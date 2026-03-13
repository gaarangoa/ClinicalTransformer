from clinical_transformer._legacy.datasets.preprocessor.tabular import Preprocessor
from clinical_transformer._legacy.datasets.dataloader.tabular import TabularDataset
from clinical_transformer._legacy.models.masked_prediction import MaskedSSL
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np 
import sys

import torch
from torch import nn
import yaml
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import lightning 

from clinical_transformer._legacy.training import Config

from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning.pytorch.strategies import DeepSpeedStrategy

from clinical_transformer._legacy.losses.survival.cindex import cindex_loss

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import logging

logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Survival(lightning.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        MasterModel = MaskedSSL.load_from_checkpoint(config.pretrained.from_pretrained)
        self.encoder = MasterModel.encoder
        self.regressor = torch.nn.Linear(MasterModel.hparams.ninp, config.model.n_classes)        
        self.loss = config.loss.name
    
        self.optimizer_class = config.optimizer.name
        self.lr = config.optimizer.params.lr

        MasterModel = None
        # self.save_hyperparameters()

    def forward(self, **kwargs):
        tokens = kwargs.get('tokens', None)
        values = kwargs.get('values', None)
        
        output_embeddings = self.encoder(
            tokens=tokens, 
            values=values
        )

        output = self.regressor(output_embeddings[:, 0, :]) # takes the [cls] embeddings as the input features
        return output, output_embeddings

    
    def training_step(self, batch, batch_idx):
        tokens, values, labels = batch
        out, emb = self.forward(
            tokens=tokens, 
            values=values
        )

        loss = self.loss(labels, out)
        self.log('train_loss', loss)
                
        return loss

    def configure_optimizers(self):
        optimizer =  self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer



def pipeline():
    
    config_file = sys.argv[1]
    config_ = yaml.safe_load(open(config_file, 'r'))
    config = Config(config_)

    config.pretrained.from_pretrained = os.path.abspath(f'{config.pretrained.path}/models/{config.pretrained.model_name}/version_{config.pretrained.version}/models/epoch={config.pretrained.epoch}.ckpt/lightning_model.pt')
    
    # Loss
    config.loss.name = eval(config.loss.name) # clinical_transformer._legacy.losses.survival.cindex.cindex_loss
    
    # Optimizer
    config.optimizer.name = eval(config.optimizer.name)

    # Global configs 
    torch.backends.cuda.enable_flash_sdp(enabled=config.model.enable_flash_attention)
    seed_everything(config.experiment.seed, workers=True)
    torch.set_float32_matmul_precision(config.experiment.set_float32_matmul_precision)
    config.experiment.output_dir = f'{config.experiment.save_dir}/{config.experiment.name}/version_{config.experiment.version}/'

    # Loggers
    csv_logger = CSVLogger(
        save_dir=config.experiment.save_dir, 
        name=config.experiment.name, 
        version=config.experiment.version
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="epoch",  # You can monitor a specific metric, but for saving after each epoch, we use "epoch"
        dirpath=f"{config.experiment.output_dir}/models/",  # Directory to save the models
        filename="{epoch}",  # Filename format for saving the model
        save_top_k=-1,  # Save all models (otherwise it saves only the best `k` models)
        save_weights_only=False,  # Save only the model weights (no optimizer states, etc.)
        every_n_epochs=config.trainer.save_every_n_epochs  # Save every epoch
    )

    # Load data
    data_processed = pickle.load(open(config.dataset.input_file, 'rb'))
    labels = pickle.load(open(config.dataset.labels_file, 'rb'))
    labels = np.array(labels)

    # Build classifier
    model = Survival(config=config)

    # Trainer
    if config.trainer.strategy.name == 'deepspeed':
        ds_strategy = DeepSpeedStrategy(
            **config.trainer.strategy.params.__dict__
        )
    else:
        ds_strategy = config.trainer.strategy.name
    
    trainer = Trainer(
        log_every_n_steps=config.trainer.log_every_n_steps,
        deterministic=config.trainer.deterministic,
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
        accelerator=config.trainer.accelerator,
        strategy=ds_strategy,
        max_epochs=config.trainer.epochs,
        precision=config.trainer.precision,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=config.trainer.reload_dataloaders_every_n_epochs,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
    )

    dataset = TabularDataset(
        data_processed, 
        labels=labels, 
        context_window=config.dataset.context_window, 
        masking_fraction=config.dataset.masking_fraction,
        mask_values=config.dataset.mask_values
    )
    train_dataloader = DataLoader(
        dataset, 
        batch_size=config.dataset.batch_size, 
        shuffle=config.dataset.shuffle, 
        num_workers=config.dataset.num_workers
    )
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        ckpt_path= config.trainer.from_checkpoint
    )

if __name__ == "__main__":
    pipeline()
