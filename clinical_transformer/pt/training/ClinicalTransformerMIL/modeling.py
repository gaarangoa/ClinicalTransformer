import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional


class GatedAttention(nn.Module):
    def __init__(self, config):
        super(GatedAttention, self).__init__()
        '''
        feature_dim: dimensionality of each instance feature vector
        hidden_dim: dimension of the hidden attention space
        output_dim: dimension of the final bag embedding
        num_attn_heads: number of independent attention heads
        '''
        self.attention_clip_prob = config.attention_clip_prob
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.num_attn_heads = config.num_attn_heads

        # Branch 1: nonlinear transformation with tanh
        self.value_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.Tanh()
        )

        # Branch 2: nonlinear transformation with sigmoid (gate)
        self.gate_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.Sigmoid()
        )

        # Final linear projection -> raw attention logits
        # self.attention_projection = OrthonormalLinear(hidden_dim, num_attn_heads, bias=False)
        self.attention_projection = nn.Linear(config.hidden_dim, config.num_attn_heads, bias=False)

        # Final projection layer to aggregate multi-head bag embeddings
        self.head_projection = nn.Linear(config.num_attn_heads * config.feature_dim, config.output_dim)
        self.layer_norm = nn.LayerNorm(config.output_dim)

    def forward(self, instance_features, mask=None):
        """
        instance_features: Tensor of shape (batch_size, num_instances, feature_dim)
        mask: Tensor of shape (batch_size, num_instances), 1 for valid instances, 0 for padding
        Returns:
            bag_embedding: Tensor of shape (batch_size, output_dim)
            attention_weights: Tensor of shape (batch_size, num_instances, num_attn_heads)
        """

        batch_size, num_instances, feature_dim = instance_features.size()

        # v_i: projected instance representation (tanh branch)
        values = self.value_projection(instance_features)  # (B, N, hidden_dim)

        # u_i: projected gating representation (sigmoid branch)
        gates = self.gate_projection(instance_features)  # (B, N, hidden_dim)

        # Element-wise gated representation g_i = v_i ⊙ u_i
        gated_features = values * gates  # (B, N, hidden_dim)

        # Raw attention logits for each instance
        attn_logits = self.attention_projection(gated_features)  # (B, N, K)

        # Apply mask if provided: set padded positions to -inf
        if mask is not None:
            # mask: (B, N) -> expand to (B, N, K)
            mask_expanded = mask.unsqueeze(-1).expand_as(attn_logits)
            attn_logits = attn_logits.masked_fill(mask_expanded == 0, -1e10)

        # Normalize across instances so that weights sum to 1 for valid instances
        attention_weights = F.softmax(attn_logits, dim=1)  # (B, N, K)

        attn_min = attention_weights.min(dim=1, keepdim=True)[0]  # (B, 1, K)
        attn_max = attention_weights.max(dim=1, keepdim=True)[0]  # (B, 1, K)
        attention_weights = (attention_weights - attn_min) / (attn_max - attn_min + 1e-8)

        attention_weights = torch.where(
            attention_weights >= self.attention_clip_prob,
            attention_weights,
            torch.zeros_like(attention_weights)
        )
        
        # Step 5: Compute bag-level embedding (weighted sum of instance features)
        expanded_features = instance_features.unsqueeze(2).repeat(1, 1, self.num_attn_heads, 1)  # (B, N, K, F)
        expanded_weights = attention_weights.unsqueeze(-1)  # (B, N, K, 1)

        # Weighted sum across instances
        bag_embeddings_per_head = torch.sum(expanded_weights * expanded_features, dim=1)  # (B, K, F)

        # Flatten heads
        bag_embedding_concat = bag_embeddings_per_head.view(batch_size, -1)  # (B, K*F)

        # Project to output_dim and apply LayerNorm
        bag_embedding = self.head_projection(bag_embedding_concat)  # (B, output_dim)
        bag_embedding = self.layer_norm(bag_embedding)

        return bag_embedding, attention_weights


@dataclass
class ClinicalTransformerMILOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None
    attentions: Optional[torch.FloatTensor] = None

class ClinicalTransformerMILConfig(PretrainedConfig):
    model_type = 'ClinicalTransformerMIL'

    def __init__(
        self, 
        bag_size = 10000,
        feature_dim = 1024,
        hidden_dim = 512,
        num_attn_heads = 8,
        output_dim = 128,
        attention_clip_prob = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.bag_size = bag_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_attn_heads = num_attn_heads
        self.output_dim = output_dim
        self.attention_clip_prob = attention_clip_prob

class ClinicalTransformerMILModel(PreTrainedModel):
    config_class = ClinicalTransformerMILConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.GatedAttentionModel = GatedAttention(config)

    def forward(self, hidden_states: torch.Tensor, mask = None, **kwargs):
        hidden_states, attentions = self.GatedAttentionModel(hidden_states, mask=mask)
        if not kwargs.get('return_attentions', False):
            attentions = ()
            
        return ClinicalTransformerMILOutput(
            hidden_states = hidden_states,
            attentions = attentions
        )