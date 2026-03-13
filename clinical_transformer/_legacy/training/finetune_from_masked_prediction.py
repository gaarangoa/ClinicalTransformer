import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import yaml 
import json

# home_dir = os.getenv("HOME")

# sys.path.append(f'{home_dir}/foundation_models/')
# sys.path.append(f'{home_dir}/samecode/')

import torch
from clinical_transformer._legacy.datasets.preprocessor.tabular import Preprocessor
from clinical_transformer._legacy.datasets.dataloader.tabular import TabularDataset
from clinical_transformer._legacy.models.masked_prediction import MaskedSSL
from clinical_transformer._legacy.models.classifier import Classifier
from clinical_transformer._legacy.models.regressor import Regressor
from clinical_transformer._legacy.training import Config

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

import logging 

logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_pretrained_model(config):
    preprocessor = Preprocessor().load(config_file=f'{config.pretrained.path}/preprocessor.yaml')
    
    if not os.path.exists(f'{config.pretrained.path}/models/epoch={config.pretrained.epoch}.ckpt/lightning_model.pt'):
        logger.info('Convert deepspeed checkpoint to single checkpoint ...')
        convert_zero_checkpoint_to_fp32_state_dict(
            f"{config.pretrained.path}models/epoch={config.pretrained.epoch}.ckpt",
            f"{config.pretrained.path}/models/epoch={config.pretrained.epoch}.ckpt/lightning_model.pt"
        );

    logger.info(f'Loading pretrained model {config.pretrained.path}/models/epoch={config.pretrained.epoch}.ckpt/lightning_model.pt')
    pretrained_model = MaskedSSL.load_from_checkpoint(f'{config.pretrained.path}/models/epoch={config.pretrained.epoch}.ckpt/lightning_model.pt').to('cpu')
    state_dict_encoder = pretrained_model.encoder.state_dict()

    pretrained_model = None
    return state_dict_encoder, preprocessor

def pipeline(config):

    # check if the task is finetunign or not
    try: 
        config.pretrained
        logger.info('Performing Finetuning ...')
        state_dict_encoder, preprocessor = load_pretrained_model(config)
        is_pretrained = True
    except:
        is_pretrained = False

    # Global configs 
    logger.info('setting global configuration flash attention, seed, matmul precision...')
    torch.backends.cuda.enable_flash_sdp(enabled=config.model.enable_flash_attention)
    seed_everything(config.experiment.seed, workers=True)
    torch.set_float32_matmul_precision(config.experiment.set_float32_matmul_precision)
    config.experiment.output_dir = f'{config.experiment.save_dir}/{config.experiment.name}/version_{config.experiment.version}/'
    logger.info(f'Experiment will be saved: {config.experiment.save_dir}/{config.experiment.name}/version_{config.experiment.version}/')

    # Loggers
    logger.info('setting up loggers and callbacks ...')
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
    logger.info(f'loading dataset {config.dataset.input_file}, {config.dataset.labels_file}')
    X_train = json.load(open(config.dataset.input_file, 'r'))
    y_train = json.load(open(config.dataset.labels_file, 'r'))

    logger.info(f'Number of samples {len(y_train)}')

    if type(config.dataset.categorical_features) == str:
        config.dataset.categorical_features = json.load(open(config.dataset.categorical_features, 'r'))
    if type(config.dataset.numerical_features) == str:
        config.dataset.numerical_features = json.load(open(config.dataset.numerical_features, 'r'))
    
    logger.info(f'Number of numerical features: {len(config.dataset.numerical_features)}')
    logger.info(f'Number of categorical features: {len(config.dataset.categorical_features)}')

    # process data with the pretrained preprocessor
    logger.info('setting up dataloader ...')

    if not is_pretrained:
        # build preprocessor
        preprocessor = Preprocessor(
            categorical_features=config.dataset.categorical_features,
            numerical_features=config.dataset.numerical_features,
            output_dir=config.experiment.output_dir
        )
        preprocessor = preprocessor.fit(X_train)
    else:
        # save preprocessor to new directory
        preprocessor.output_dir = config.experiment.output_dir
        preprocessor.save()

    X_train_processed = preprocessor.transform(X_train, context_window=config.dataset.context_window)

    train_dataset = TabularDataset(
        X_train_processed, 
        y_train,
        context_window=config.dataset.context_window, 
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.dataset.batch_size, 
        shuffle=config.dataset.shuffle, 
        num_workers=config.dataset.num_workers
    )

    # Model
    logger.info('setting up model ...')
    if config.model.task == 'classification':
        model_class = Classifier
    if config.model.task == 'regression': 
        model_class = Regressor
    if config.model.task == 'survival':
        model_class = Regressor
    if config.model.task == 'pbmf':
        model_class = Classifier
    
    logger.info(f'setting up model task: {config.model.task}')
    
    model = model_class(
        ntoken=config.model.vocabulary_size, # Vocabulary size
        ninp=config.model.embedding_size, # Embedding size
        nhid=config.model.hidden_layer_size, # FFN layer size
        nclasses=config.model.classes,
        nhead=config.model.heads, 
        nlayers=config.model.layers, 
        batch_first=True,
        output_dir=config.experiment.output_dir,
        lr=config.optimizer.lr,
        dropout=config.model.dropout
    )
    
    if is_pretrained:
        # transfer encoder weights
        logger.info(f'Transfer learning: Initializing model with pretrained weights ...')
        model.encoder.load_state_dict(state_dict_encoder)
        state_dict_encoder = None

    logger.info('setting up trainer ...')
    trainer = Trainer(
        log_every_n_steps=config.trainer.log_every_n_steps,
        deterministic=config.trainer.deterministic,
        devices=config.trainer.devices,
        accelerator=config.trainer.accelerator,
        strategy=config.trainer.strategy,
        max_epochs=config.trainer.epochs,
        # precision=config.trainer.precision,
        reload_dataloaders_every_n_epochs=config.trainer.reload_dataloaders_every_n_epochs,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        ckpt_path= config.trainer.from_checkpoint
    )


if __name__ == "__main__":
    # Load config file
    config_file = sys.argv[1]
    config_ = yaml.safe_load(open(config_file, 'r'))
    config = Config(config_)
    
    pipeline(config=config)

