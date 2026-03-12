import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

@dataclass
class VanillaVAEOutput(ModelOutput):
    """
    Output class for VanillaVAE model.
    """
    reconstruction: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
    z: Optional[torch.Tensor] = None


class Encoder(nn.Module):
    """
    VAE Encoder that maps input to latent mean and log variance.
    """
    
    def __init__(self, config):
        """
        Initialize the encoder.
        
        Args:
            config: VanillaVAEConfig containing model configuration
        """
        super(Encoder, self).__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, config.latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (mean, log_variance) tensors
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    VAE Decoder that maps latent code back to data space.
    """
    
    def __init__(self, config):
        """
        Initialize the decoder.
        
        Args:
            config: VanillaVAEConfig containing model configuration
        """
        super(Decoder, self).__init__()
        
        # Build decoder layers (reverse hidden dims for decoder)
        layers = []
        prev_dim = config.latent_dim
        decoder_hidden_dims = config.hidden_dims[::-1]
        
        for hidden_dim in decoder_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, config.input_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed output tensor
        """
        return self.decoder(z)


class VanillaVAEConfig(PretrainedConfig):
    """
    Configuration class for VanillaVAE model.
    """
    model_type = 'VanillaVAE'

    def __init__(
        self,
        input_dim: int = 2000,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 64,
        beta: float = 1.0,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.beta = beta
        self.dropout_rate = dropout_rate


class VanillaVAE(PreTrainedModel):
    config_class = VanillaVAEConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim
        self.beta = config.beta
        
        # Initialize encoder and decoder
        self.encoder = Encoder(
            config
        )
        self.decoder = Decoder(
            config
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        
        Args:
            mu: Mean tensor
            logvar: Log variance tensor
            
        Returns:
            Sampled latent tensor
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            VanillaVAEOutput containing:
                - reconstruction: Reconstructed input
                - mu: Latent mean
                - logvar: Latent log variance
                - z: Sampled latent code
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)

        return VanillaVAEOutput(
            reconstruction=reconstruction,
            mu=mu,
            logvar=logvar,
            z=z
        )
