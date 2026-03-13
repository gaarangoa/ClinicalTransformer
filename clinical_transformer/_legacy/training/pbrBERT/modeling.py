import sys
import os
import yaml 

import lightning
import torch
from transformers.models.bert.modeling_bert import BertLayer, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput
from clinical_transformer.pt.training.pbrBERT.dataset import MaskedTokenDataset as TabularMaskedDataset

from clinical_transformer.pt.losses.masked_prediction import MaskPredictionLoss
from clinical_transformer.pt.training import Config

import pickle

from torch.utils.data import DataLoader
from lightning.pytorch import Trainer

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader
import logging 

from lightning.pytorch.strategies import DeepSpeedStrategy

from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam

from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class nBERTModelOutput(ModelOutput):
    """
    Output class for nBERT model following HuggingFace Transformers pattern.
    
    Args:
        last_hidden_state (torch.FloatTensor): Sequence of hidden-states at the output of the last layer.
            Shape: (batch_size, sequence_length, hidden_size)
        token_predictions (torch.FloatTensor): Prediction scores for token reconstruction.
            Shape: (batch_size, sequence_length, vocab_size)
        value_predictions (torch.FloatTensor, optional): Prediction scores for value reconstruction.
            Shape: (batch_size, sequence_length)
        hidden_states (tuple, optional): Hidden-states of the model at the output of each layer.
            Tuple of torch.FloatTensor, one for each layer.
        attentions (tuple, optional): Attention weights after the attention softmax.
            Tuple of torch.FloatTensor, one for each layer.
        input_embeddings (torch.FloatTensor, optional): Input embeddings before passing through encoder.
            Shape: (batch_size, sequence_length, hidden_size)
    """
    last_hidden_state: torch.FloatTensor = None
    token_predictions: torch.FloatTensor = None
    value_predictions: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    input_embeddings: Optional[torch.FloatTensor] = None


class CTEmbeddings(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.value_embeddings = torch.nn.Linear(in_features=1, out_features=config.hidden_size, bias=True)
        self.scaling = torch.sqrt(torch.tensor(config.hidden_size))
        
        self.token_ln = torch.nn.LayerNorm(config.hidden_size)
        self.value_ln = torch.nn.LayerNorm(config.hidden_size)
        self.final_ln = torch.nn.LayerNorm(config.hidden_size)

    def forward(self, **kwargs):
        # padding mask
        # - 1 for tokens that are **not masked**,
        # - 0 for tokens that are **masked**.

        tokens = kwargs.get('tokens', None)
        values = kwargs.get('values', None)
        
        values = values.unsqueeze(-1)
        padding_mask = (tokens != 0)

        token_embeddings = self.token_embeddings(tokens)
        value_embeddings = self.value_embeddings(values)

        token_emb = self.token_ln(token_embeddings)
        value_emb = self.value_ln(value_embeddings)

        embeddings = (token_emb + value_emb) * self.scaling
        embeddings = self.final_ln(embeddings)

        return embeddings, padding_mask

class nBERTPretrainedModel(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # INPUT
        self.embedder = CTEmbeddings(config)

        # ENCODER TRANSFORMER
        self.encoder = torch.nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.hidden_size = config.hidden_size
        self.output_ln = torch.nn.LayerNorm(config.hidden_size)
        
        self.post_init()
    
    def get_input_embeddings(self):
        """Get the token embeddings layer."""
        return self.embedder.token_embeddings
    
    def set_input_embeddings(self, value):
        """Set the token embeddings layer."""
        self.embedder.token_embeddings = value
    
    def _init_weights(self, module):
        """Initialize weights following BERT initialization."""
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
         
    def forward(self, tokens, values, **kwargs):
        """
        Forward pass through the nBERT model.
        
        Args:
            tokens (torch.LongTensor): Input token ids. Shape: (batch_size, sequence_length)
            values (torch.FloatTensor): Input values corresponding to tokens. Shape: (batch_size, sequence_length)
            output_hidden_states (bool, optional): Whether to return hidden states for all layers.
            output_attentions (bool, optional): Whether to return attention weights for all layers.
            output_predictions (bool, optional): Whether to return token and value predictions.
            return_dict (bool, optional): Whether to return a ModelOutput instead of tuple.
            output_last_hidden_state (bool, optional): Whether to return the last hidden state.

        Returns:
            nBERTModelOutput or tuple: Model outputs including predictions and optional hidden states/attentions.
        """
        return_dict = kwargs.get('return_dict', True)
        output_hidden_states = kwargs.get('output_hidden_states', False)
        output_attentions = kwargs.get('output_attentions', False)
        output_predictions = kwargs.get('output_predictions', True)
        output_last_hidden_state = kwargs.get('output_last_hidden_state', True)
        
        # Initialize storage for optional outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Get embeddings and padding mask
        embeddings, padding_mask = self.embedder(tokens=tokens, values=values)
        embeddings = padding_mask.unsqueeze(-1).type_as(embeddings) * embeddings
        
        # Store input embeddings if requested
        input_embeddings = embeddings.detach() if kwargs.get('output_input_embeddings', False) else None
        
        # Pass through transformer layers
        hidden_state = embeddings
        for layer in self.encoder:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            
            layer_outputs = layer(
                hidden_states=hidden_state,
                attention_mask=(-1e6 * (~padding_mask).unsqueeze(1).unsqueeze(2)).type_as(hidden_state),
                output_attentions=output_attentions
            )

            if output_attentions:
                hidden_state, attention_weights = layer_outputs
                all_attentions = all_attentions + (attention_weights,)
            else:
                hidden_state = layer_outputs[0]
            
            # Zero out padding positions
            hidden_state = padding_mask.unsqueeze(-1).type_as(hidden_state) * hidden_state

        # Final layer normalization
        hidden_state = self.output_ln(hidden_state)
        hidden_state = padding_mask.unsqueeze(-1).type_as(hidden_state) * hidden_state
        
        # Add final hidden state if collecting all states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        # Generate token predictions
        token_predictions = torch.matmul(hidden_state, self.embedder.token_embeddings.weight.transpose(0, 1))
        
        # Generate value predictions (if needed in future)
        value_predictions = None  # Can be implemented later if needed
        
        if not return_dict:
            outputs = (hidden_state, token_predictions)
            if value_predictions is not None:
                outputs = outputs + (value_predictions,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs

        return nBERTModelOutput(
            last_hidden_state=hidden_state if output_last_hidden_state else (),
            token_predictions=token_predictions if output_predictions else (),
            value_predictions=value_predictions,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            input_embeddings=input_embeddings,
        )