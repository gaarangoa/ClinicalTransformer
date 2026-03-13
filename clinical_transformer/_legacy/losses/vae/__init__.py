import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict

def loss_function(
        x: torch.Tensor, 
        outputs: Dict[str, torch.Tensor],
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            x: Original input
            outputs: Dictionary from forward pass
            
        Returns:
            Dictionary containing loss components
        """
        reconstruction = outputs['reconstruction']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstruction, x, reduction='sum')
        
        # KL divergence loss
        # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (μ² + σ² - log(σ²) - 1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = reconstruction_loss + beta * kl_loss
        
        return total_loss