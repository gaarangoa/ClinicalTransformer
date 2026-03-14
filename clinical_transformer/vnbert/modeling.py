import sys
import os
import yaml 

import lightning
import torch
from transformers.models.bert.modeling_bert import BertLayer, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput

from clinical_transformer.vnbert.dataset import MaskedTokenDataset
from clinical_transformer.utils.config import Config

import pickle
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

logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass
class nBERTModelOutput(ModelOutput):
    """
    Output class for nBERT model following HuggingFace Transformers pattern.
    
    Args:
        last_hidden_state (torch.FloatTensor): Sequence of hidden-states at the output of the last layer.
            Shape: (batch_size, sequence_length, hidden_size)
        value_predictions (torch.FloatTensor): Prediction scores for value reconstruction.
            Shape: (batch_size, sequence_length)
        hidden_states (tuple, optional): Hidden-states of the model at the output of each layer.
            Tuple of torch.FloatTensor, one for each layer.
        attentions (tuple, optional): Attention weights after the attention softmax.
            Tuple of torch.FloatTensor, one for each layer.
        input_embeddings (torch.FloatTensor, optional): Input embeddings before passing through encoder.
            Shape: (batch_size, sequence_length, hidden_size)
    """
    last_hidden_state: torch.FloatTensor = None
    value_predictions: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    input_embeddings: Optional[torch.FloatTensor] = None


class CTEmbeddings(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.value_embeddings = torch.nn.Linear(in_features=1, out_features=config.hidden_size, bias=True)
        self.scaling = config.hidden_size ** 0.5
        
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

    def forward(self, tokens, values, **kwargs):
        """
        Forward pass through the base nBERT model.
        
        Args:
            tokens (torch.LongTensor): Input token ids. 
                Shape: (batch_size, sequence_length)
            values (torch.FloatTensor): Input values corresponding to tokens. 
                Shape: (batch_size, sequence_length)
            output_hidden_states (bool, optional): Whether to return hidden 
                states for all layers.
            output_attentions (bool, optional): Whether to return attention 
                weights for all layers.
            return_dict (bool, optional): Whether to return a ModelOutput 
                instead of tuple.
            output_last_hidden_state (bool, optional): Whether to return 
                the last hidden state.

        Returns:
            BaseModelOutput or tuple: Model outputs including last hidden 
                state and optional hidden states/attentions.
        """
        return_dict = kwargs.get('return_dict', True)
        output_hidden_states = kwargs.get('output_hidden_states', False)
        output_attentions = kwargs.get('output_attentions', False)
        output_last_hidden_state = kwargs.get('output_last_hidden_state', 
                                              True)
        
        # Initialize storage for optional outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Create masks for different token types using scGPT strategy
        padding_mask = (tokens != 0)  # Not padding tokens
        masked_positions = (values == -10.0)  # Masked value positions

        # scGPT attention strategy: masked positions excluded from keys/values
        # but can still be queries (can attend to others AND themselves)
        # key_value_mask: True for positions that can be keys/values
        key_value_mask = padding_mask & (~masked_positions)

        # Zero out masked values for embedding without cloning
        values_for_embedding = torch.where(
            masked_positions, torch.zeros_like(values), values
        )

        # Get embeddings and apply ONLY padding mask (not masked positions)
        embeddings, _ = self.embedder(tokens=tokens,
                                      values=values_for_embedding)

        # Precompute padding expansion once for reuse across layers
        padding_expanded = padding_mask.unsqueeze(-1).type_as(embeddings)
        embeddings = padding_expanded * embeddings

        # Build attention mask once (identical across all layers)
        batch_size, seq_len = tokens.shape

        # Base attention mask for padding
        attention_mask = padding_mask.unsqueeze(1).unsqueeze(2).expand(
            batch_size, 1, seq_len, seq_len
        )

        # Restrict: no position can attend TO masked positions
        key_value_expanded = key_value_mask.unsqueeze(1).unsqueeze(1).expand(
            batch_size, 1, seq_len, seq_len
        )
        attention_mask = attention_mask & key_value_expanded

        # Allow masked positions to attend to themselves (diagonal)
        diagonal_mask = torch.eye(
            seq_len, device=tokens.device, dtype=torch.bool
        )
        masked_self_attend = (
            masked_positions.unsqueeze(1) & diagonal_mask.unsqueeze(0)
        )
        attention_mask = attention_mask | masked_self_attend.unsqueeze(1)

        # Convert to attention scores format (large negative for blocked)
        attention_mask = (-1e6 * (~attention_mask)).type_as(embeddings)

        # Pass through transformer layers
        hidden_state = embeddings
        for layer in self.encoder:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer(
                hidden_states=hidden_state,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )

            if output_attentions:
                hidden_state, attention_weights = layer_outputs
                all_attentions = all_attentions + (attention_weights,)
            else:
                hidden_state = layer_outputs[0]

            # Zero out padding positions
            hidden_state = padding_expanded * hidden_state

        # Final layer normalization
        hidden_state = self.output_ln(hidden_state)
        hidden_state = padding_expanded * hidden_state
        
        # Add final hidden state if collecting all states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        # For backward compatibility, handle legacy kwargs
        if kwargs.get('output_last_states', False):
            output_last_hidden_state = True
        
        if not return_dict:
            outputs = (hidden_state,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs

        return BaseModelOutput(
            last_hidden_state=(hidden_state if output_last_hidden_state
                               else None),
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class nBertPretrainedModelForMaskingValuePrediction(nBERTPretrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        # Value prediction head
        self.value_predictor = torch.nn.Linear(config.hidden_size, 1)
        
        self.post_init()

    def forward(self, tokens, values, **kwargs):
        """
        Forward pass through the nBERT model for masking value prediction.
        
        Args:
            tokens (torch.LongTensor): Input token ids. 
                Shape: (batch_size, sequence_length)
            values (torch.FloatTensor): Input values corresponding to tokens. 
                Shape: (batch_size, sequence_length)
            output_hidden_states (bool, optional): Whether to return hidden 
                states for all layers.
            output_attentions (bool, optional): Whether to return attention 
                weights for all layers.
            output_predictions (bool, optional): Whether to return value 
                predictions.
            return_dict (bool, optional): Whether to return a ModelOutput 
                instead of tuple.
            output_last_hidden_state (bool, optional): Whether to return 
                the last hidden state.

        Returns:
            nBERTModelOutput or tuple: Model outputs including predictions 
                and optional hidden states/attentions.
        """
        return_dict = kwargs.get('return_dict', True)
        output_predictions = kwargs.get('output_predictions', True)
        
        # Get base model outputs
        outputs = super().forward(
            tokens=tokens,
            values=values,
            **kwargs
        )
        
        # Extract hidden state
        if return_dict:
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs[0]

        # Generate value predictions
        value_predictions = self.value_predictor(hidden_state).squeeze(-1)
        
        if not return_dict:
            return outputs + (value_predictions,)

        return nBERTModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            value_predictions=(value_predictions if output_predictions else None),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            input_embeddings=kwargs.get('output_input_embeddings', False),
        )


class LightningTrainerModel(lightning.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        # Use the specialized model for masking value prediction
        self.model = nBertPretrainedModelForMaskingValuePrediction(config)
        
        logger.info('saving hyperparamters')
        self.save_hyperparameters()
        
        self.loss = torch.nn.MSELoss(reduction='none')

        _OPTIMIZERS = {
            'torch.optim.Adam': torch.optim.Adam,
            'torch.optim.AdamW': torch.optim.AdamW,
            'DeepSpeedCPUAdam': DeepSpeedCPUAdam,
            'FusedAdam': FusedAdam,
        }
        if config.optimizer not in _OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer '{config.optimizer}'. "
                f"Supported: {list(_OPTIMIZERS.keys())}"
            )
        self.optimizer_ = _OPTIMIZERS[config.optimizer]
        self.lr = config.learning_rate
         
        
    def forward(self, **kwargs):
        out = self.model(**kwargs)

        return out

    def training_step(self, batch, batch_idx):
        tokens = batch['tokens']  # Ensure Long dtype
        values = batch['values']  # Values with -10 at masked positions
        labels = batch['labels']  # Original values (ground truth)

        out = self.forward(
            tokens=tokens,
            values=values,
            output_last_hidden_state=True,
            return_dict=True
        )

        # Create mask for masked positions only (where input values == -10)
        masked_positions = (values == -10.0)
        
        # Only compute loss for masked positions
        if masked_positions.any():
            value_pred = out.value_predictions[masked_positions]
            value_true = labels[masked_positions]
            
            # Calculate MSE loss only for masked positions
            loss = self.loss(value_pred, value_true).mean()
        else:
            # No masked positions in this batch
            loss = torch.tensor(0.0, device=values.device, requires_grad=True)
        
        self.log('train_loss', loss)
        
        return loss

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = self.optimizer_(self.parameters(), lr=self.lr)        
        return optimizer  

