import torch
import torch.nn as nn

import sys
import os
import yaml 

import lightning
import torch
from transformers import ModernBertForMaskedLM, ModernBertConfig
from clinical_transformer._legacy.datasets.preprocessor.tabular_gpt import PreprocessorGPT as Tokenizer
from clinical_transformer._legacy.datasets.dataloader.tabular_sorted_masked import MaskedTokenDataset as TabularMaskedDataset
from clinical_transformer._legacy.training import Config

import pickle
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer

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


class LightningTrainerModel(lightning.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        # INPUT
        self.model = ModernBertForMaskedLM(config)
        
        logger.info('saving hyperparamters')
        self.save_hyperparameters()
        
        self.crossentropy_loss = nn.CrossEntropyLoss()
        self.optimizer_ = eval(config.optimizer)
        self.lr = config.learning_rate
         
        
    def forward(self, **kwargs):
        out = self.model(**kwargs)

        return out

    def training_step(self, batch, batch_idx):
        tokens = batch['tokens']  # Ensure Long dtype
        labels = batch['original_tokens'].clone()       # [B, T]

        # Only compute loss on MASK tokens (set others to -100)
        labels[tokens != 1] = -100
    
        # Optional but recommended: attention mask for padding
        attention_mask = (tokens != 0).long()
    
        # Forward pass — HuggingFace will compute loss only on masked positions
        output = self.model(input_ids=tokens, attention_mask=attention_mask, labels=labels)
        loss = output.loss

        self.log('train_loss', loss)
        
        return loss

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = self.optimizer_(self.parameters(), lr=self.lr)
        
        return optimizer  

def pipeline():
    """
    Pipeline for training BERT model using clinical transformer.
    """
    config_file = sys.argv[1]
    config_ = yaml.safe_load(open(config_file, 'r'))
    config = Config(config_)

    # Global configs 
    torch.backends.cuda.enable_flash_sdp(enabled=config.model.enable_flash_attention)
    seed_everything(config.experiment.seed, workers=True)
    torch.set_float32_matmul_precision(config.experiment.set_float32_matmul_precision)
    config.experiment.output_dir = f'{config.experiment.save_dir}/{config.experiment.name}/version_{config.experiment.version}/'
    os.makedirs(config.experiment.output_dir, exist_ok=True)

    yaml.dump(config_, open(f'{config.experiment.output_dir}/experiment_{config_file.split("/")[-1]}', 'w'))
    
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
    except:
        logger.error('input file has to be the pre-processed file in pickle format. Loading json files is deprecated.')
        sys.exit(1)

    train_dataset = TabularMaskedDataset(
        X_train_processed, 
        context_window=config.dataset.context_window, 
        masking_fraction=config.dataset.masking_fraction,
        mask_values=config.dataset.mask_values,
        return_values=False,
        return_cls=True
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.dataset.batch_size, 
        shuffle=config.dataset.shuffle, 
        num_workers=config.dataset.num_workers
    )

    model_config = ModernBertConfig(
        **config.model.__dict__,
    )

    model = LightningTrainerModel(model_config)
    model_config.to_json_file(f'{config.experiment.output_dir}/model_config.json')
    
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