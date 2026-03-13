import torch
from torch import nn
import yaml
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import lightning as L

from .core import BaseClinicalTransformer

class Classifier(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        n_classes = kwargs.get('nclasses', None)
        ninp = kwargs.get('ninp', 128)
        self.lr = kwargs.get('lr', 1e-3)

        self.encoder = BaseClinicalTransformer(**kwargs)
        self.decoder = nn.Linear(ninp, n_classes)

        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.save_hyperparameters()

    def forward(self, **kwargs):
        tokens = kwargs.get('tokens', None)
        values = kwargs.get('values', None)
        
        output_embeddings = self.encoder(
            tokens=tokens, 
            values=values
        )

        output = self.decoder(output_embeddings)
        output = output[:, 0, :] # Takes the first token embeddings that should be the <cls> token

        return output, output_embeddings

    def training_step(self, batch, batch_idx):
        tokens, values, labels = batch
        out, emb = self.forward(
            tokens=tokens, 
            values=values
        )

        loss = self.loss(out, labels)
        auc = self.compute_auc(out, labels)
        
        self.log('train_loss', loss)
        self.log('train_auc', auc)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_auc(self, y_pred, y_true):
        y_pred = y_pred.detach().cpu().numpy()[:, 0]
        y_true = y_true.detach().cpu().numpy()
        
        # Compute AUC (for binary classification, use the true positive rate)
        return roc_auc_score(y_true, y_pred)