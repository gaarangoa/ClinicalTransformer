import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import yaml 
import json
import pickle

import torch
from clinical_transformer._legacy.datasets.preprocessor.tabular import Preprocessor
from clinical_transformer._legacy.datasets.dataloader.tabular import TabularMaskedDataset
from clinical_transformer._legacy.models.masked_prediction import MaskedSSL
from clinical_transformer._legacy.training import Config

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader
import logging 

from lightning.pytorch.strategies import DeepSpeedStrategy

from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam

logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def pipeline():
    """
    The pipeline function sets up and executes the training process for a machine learning model.
    It performs the following steps:
    1. Loads the configuration file.
    2. Sets global configurations such as enabling flash attention, setting random seed, and float32 matmul precision.
    3. Initializes loggers and model checkpointing.
    4. Loads preprocessed training data.
    5. Prepares the training dataset.
    6. Setup the model 
    7. Trains the model using the trainer class
    """
    config_file = sys.argv[1]
    config_ = yaml.safe_load(open(config_file, 'r'))
    config = Config(config_)

    # try:
    #     config.trainer.devices = sys.argv[2]
    #     config.trainer.num_nodes = sys.argv[3]
    #     logger.info(f'Redefining {config.trainer.dpevices} devices and {config.trainer.num_nodes} nodes')
    # except:
    #     pass

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

    # Dataset
    try:
        logger.info('loading preprocessed data ...') 
        X_train_processed = pickle.load(open(config.dataset.input_file, 'rb'))
        logger.warning('building preprocessor no longer supported, please do it prior to running this script ...')
    except:
        logger.error('input file has to be the pre-processed file in pickle format. Loading json files is deprecated.')
        sys.exit(1)

    train_dataset = TabularMaskedDataset(
        X_train_processed, 
        context_window=config.dataset.context_window, 
        masking_fraction=config.dataset.masking_fraction,
        mask_values=config.dataset.mask_values
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.dataset.batch_size, 
        shuffle=config.dataset.shuffle, 
        num_workers=config.dataset.num_workers
    )

    # Model
    model = MaskedSSL(
        ntoken=config.model.vocabulary_size, # Vocabulary size
        ninp=config.model.embedding_size, # Embedding size
        nhid=config.model.hidden_layer_size, # FFN layer size
        nhead=config.model.heads, 
        nlayers=config.model.layers, 
        batch_first=True,
        output_dir=config.experiment.output_dir,
        lr=config.optimizer.params.lr,
        loss_token_weight=config.loss.loss_token_weight,
        loss_value_weight=config.loss.loss_value_weight,
        dropout=config.model.dropout,
        optimizer=eval(config.optimizer.name)
    )
    
    if config.trainer.strategy['name'] == 'deepspeed':
        ds_strategy = DeepSpeedStrategy(
            **config.trainer.strategy.params.__dict__
        )
    else:
        ds_stragety = config.trainer.strategy['name']
    
    # Trainer
    trainer = Trainer(
        log_every_n_steps=config.trainer.log_every_n_steps,
        deterministic=config.trainer.deterministic,
        devices=config.trainer.devices,
        accelerator=config.trainer.accelerator,
        strategy=ds_strategy,
        max_epochs=config.trainer.epochs,
        precision=config.trainer.precision,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=config.trainer.reload_dataloaders_every_n_epochs,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
        num_nodes=config.trainer.num_nodes,
    )

    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        ckpt_path= config.trainer.from_checkpoint
    )

if __name__ == "__main__":
    pipeline()