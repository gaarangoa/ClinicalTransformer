import torch
from torch import nn
import yaml
import torch.nn.functional as F
import lightning as L

from .core import BaseClinicalTransformer
from clinical_transformer.pt.losses.masked_prediction import MaskPredictionLoss

from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam


class MaskedSSL(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        ntoken = kwargs.get('ntoken', 1024)
        ninp = kwargs.get('ninp', 128)
        loss_tw = kwargs.get('loss_token_weight', 1)
        loss_vw = kwargs.get('loss_value_weight', 1)
        self.lr = kwargs.get('lr', 1e-3)

        self.encoder = BaseClinicalTransformer(**kwargs)
        self.token_decoder = nn.Linear(ninp, ntoken)
        self.value_decoder = nn.Linear(ninp, 1)
        
        self.loss = MaskPredictionLoss(token_weight=loss_tw, value_weight=loss_vw)
        self.save_hyperparameters()
        self.optimizer_ = kwargs.get('optimizer', torch.optim.Adam)

    def forward(self, **kwargs):
        tokens = kwargs.get('tokens', None)
        values = kwargs.get('values', None)
        
        output_embeddings = self.encoder(
            tokens=tokens, 
            values=values
        )

        token_output = self.token_decoder(output_embeddings)
        # the loss reveices logits, therefore this is not needed.
        # token_output = torch.nn.functional.softmax(token_output, dim=-1) # Takes the first token embeddings that should be the <cls> token
        value_output = self.value_decoder(output_embeddings)

        return [token_output, value_output], output_embeddings

    def training_step(self, batch, batch_idx):
        tokens, values, [token_labels, value_labels] = batch

        [token_out, value_out], _ = self.forward(
            tokens=tokens, 
            values=values
        )

        loss = self.loss(tokens, token_out, value_out, token_labels, value_labels)
        
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = self.optimizer_(self.parameters(), lr=self.lr)        
        return optimizer