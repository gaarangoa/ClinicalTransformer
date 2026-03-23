import sys
import os
import yaml

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput

from clinical_transformer.dbert.dataset import MaskedTokenDataset
from clinical_transformer.dbert.config import Config

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


# ---------------------------------------------------------------------------
# Custom transformer layer using PyTorch SDPA directly (bypasses HuggingFace)
# ---------------------------------------------------------------------------

class SDPAttention(nn.Module):
    """Multi-head attention using F.scaled_dot_product_attention directly.

    Supports arbitrary float attention masks (e.g. scGPT 4D mask) without
    going through HuggingFace's sdpa_attention_forward wrapper.
    SDPA auto-selects the best backend: flash (no mask), memory-efficient
    (with mask), or math fallback.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.attention_probs_dropout_prob

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Qwen-style per-head sigmoid gate after SDPA (Qiu et al., 2025)
        # Gate is query-dependent: g = sigmoid(W_g @ q), applied per head
        self.gated_attention = getattr(config, 'gated_attention', False)
        if self.gated_attention:
            self.gate_proj = nn.Linear(self.head_dim, self.head_dim)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (B, S, H)
            attention_mask: None, or float tensor broadcastable to (B, num_heads, S, S)
                            with 0 for attend and -inf for block.
        Returns:
            (B, S, H)
        """
        B, S, _ = hidden_states.shape

        # Project and reshape: (B, S, H) -> (B, num_heads, S, head_dim)
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA: auto-selects flash (no mask) or memory-efficient (with mask)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Per-head sigmoid gate: query-dependent sparsity on SDPA output
        if self.gated_attention:
            gate = torch.sigmoid(self.gate_proj(q))  # (B, num_heads, S, head_dim)
            attn_out = attn_out * gate

        # Reshape back: (B, num_heads, S, head_dim) -> (B, S, H)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(attn_out)



class SDPTransformerLayer(nn.Module):
    """Transformer layer: SDPA attention + FFN with post-LN residuals."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = SDPAttention(config)
        self.attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ffn_dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_act = nn.GELU()
        self.ffn_dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Self-attention with post-LN residual
        attn_out = self.attention(hidden_states, attention_mask)
        hidden_states = self.attn_ln(hidden_states + self.attn_dropout(attn_out))

        # FFN with post-LN residual
        ffn_out = self.ffn_dense_out(self.ffn_act(self.ffn_dense_in(hidden_states)))
        hidden_states = self.ffn_ln(hidden_states + self.ffn_dropout(ffn_out))

        return (hidden_states,)



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

        # ENCODER TRANSFORMER (custom SDPA layers, bypass HuggingFace wrapper)
        self.encoder = nn.ModuleList(
            [SDPTransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.hidden_size = config.hidden_size
        self.output_ln = torch.nn.LayerNorm(config.hidden_size)
        self.mask1_ratio = getattr(config, 'mask1_ratio', 0.5)
        self.mask1_target = self.mask1_ratio
        self.mask1_warmup_epochs = getattr(config, 'mask1_warmup_epochs', 0)

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

        # Create masks for different token types
        padding_mask = (tokens != 0)  # Not padding tokens
        any_masked = (values == -10.0)  # All masked positions

        # Zero out masked values for embedding without cloning
        values_for_embedding = torch.where(
            any_masked, torch.zeros_like(values), values
        )

        # Get embeddings and apply ONLY padding mask (not masked positions)
        embeddings, _ = self.embedder(tokens=tokens,
                                      values=values_for_embedding)

        batch_size, seq_len = tokens.shape
        padding_expanded = padding_mask.unsqueeze(-1).type_as(embeddings)
        embeddings = padding_expanded * embeddings

        # Build attention mask based on mask1_ratio:
        #   mask1_ratio = -1.0  → BERT:  all-to-all (padding only)
        #   mask1_ratio =  0.0  → scGPT: masked see known + self
        #   mask1_ratio >  0.0  → dBERT: split into isolated + contextual

        if self.mask1_ratio < 0:
            # BERT mode: all non-padding positions attend to all non-padding
            if padding_mask.all():
                attention_mask = None
            else:
                attention_mask = padding_mask.unsqueeze(1).unsqueeze(2).expand(
                    batch_size, 1, seq_len, seq_len
                )
                attention_mask = (torch.finfo(embeddings.dtype).min * (~attention_mask)).type_as(embeddings)
        else:
            # scGPT / dBERT mode

            # Split masked positions into isolated (mask1) and contextual (mask2)
            masked1_positions = torch.zeros_like(any_masked)
            if self.mask1_ratio > 0:
                for b in range(tokens.shape[0]):
                    masked_indices = torch.where(any_masked[b])[0]
                    n_mask1 = int(len(masked_indices) * self.mask1_ratio)
                    if n_mask1 > 0:
                        masked1_positions[b, masked_indices[:n_mask1]] = True

            # Key-value mask: only known (non-masked) positions can be keys
            key_value_mask = padding_mask & (~any_masked)

            # Base: all queries can attend to known keys
            attention_mask = padding_mask.unsqueeze(1).unsqueeze(2).expand(
                batch_size, 1, seq_len, seq_len
            )
            key_value_expanded = key_value_mask.unsqueeze(1).unsqueeze(1).expand(
                batch_size, 1, seq_len, seq_len
            )
            attention_mask = attention_mask & key_value_expanded

            # Block masked1 query rows entirely (they cannot see known tokens)
            if self.mask1_ratio > 0:
                not_masked1 = (~masked1_positions).unsqueeze(-1).unsqueeze(1).expand(
                    batch_size, 1, seq_len, seq_len
                )
                attention_mask = attention_mask & not_masked1

            # Add diagonal self-loop for ALL masked positions
            diagonal_mask = torch.eye(
                seq_len, device=tokens.device, dtype=torch.bool
            )
            masked_self_attend = (
                any_masked.unsqueeze(1) & diagonal_mask.unsqueeze(0)
            )
            attention_mask = attention_mask | masked_self_attend.unsqueeze(1)

            attention_mask = (torch.finfo(embeddings.dtype).min * (~attention_mask)).type_as(embeddings)

        hidden_state = embeddings
        for layer in self.encoder:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            hidden_state = layer(hidden_state, attention_mask=attention_mask)[0]
            hidden_state = padding_expanded * hidden_state

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

        # Generate value predictions (skip if caller will handle it)
        if output_predictions:
            value_predictions = self.value_predictor(hidden_state).squeeze(-1)
        else:
            value_predictions = None

        if not return_dict:
            return outputs + (value_predictions,)

        return nBERTModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            value_predictions=(value_predictions if output_predictions else None),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            input_embeddings=None,
        )


class LightningTrainerModel(lightning.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
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


    def on_train_epoch_start(self):
        """Anneal mask1_ratio from 0 to target over warmup epochs."""
        warmup = self.model.mask1_warmup_epochs
        if warmup > 0:
            progress = min(self.current_epoch / warmup, 1.0)
            self.model.mask1_ratio = progress * self.model.mask1_target

    def forward(self, **kwargs):
        out = self.model(**kwargs)

        return out

    def training_step(self, batch, batch_idx):
        tokens = batch['tokens']
        values = batch['values']
        labels = batch['labels']

        masked_positions = (values == -10.0)

        if not masked_positions.any():
            self.log('train_loss', 0.0)
            return torch.tensor(0.0, device=values.device, requires_grad=True)

        # Get hidden states without running prediction head on all positions
        out = self.forward(
            tokens=tokens,
            values=values,
            output_last_hidden_state=True,
            output_predictions=False,
            return_dict=True
        )

        # Run prediction head only on masked positions (both types)
        masked_hidden = out.last_hidden_state[masked_positions]
        value_pred = self.model.value_predictor(masked_hidden).squeeze(-1)
        value_true = labels[masked_positions]

        loss = self.loss(value_pred, value_true).mean()
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = self.optimizer_(self.parameters(), lr=self.lr)
        return optimizer

