import sys
import os
import yaml 

import lightning
import torch
from transformers.models.bert.modeling_bert import BertLayer, BertConfig
from clinical_transformer._legacy.losses.masked_prediction import MaskPredictionLoss
from clinical_transformer._legacy.training import Config

import pickle
from clinical_transformer._legacy.datasets.dataloader.tabular import TabularMaskedDataset
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

class CTOutput():
    def __init__(self, ):
        self.token_pred = None
        self.value_pred = None
        self.last_hidden_state = None
        self.last_attention = None
        self.input_embeddings = None
        self.hidden_states = []
        self.attentions = []

class CTEmbeddings(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.value_embeddings = torch.nn.Linear(in_features=1, out_features=config.hidden_size, bias=False)
        self.scaling = torch.sqrt(torch.tensor(config.hidden_size))

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

        embeddings = (token_embeddings + value_embeddings) * self.scaling

        return embeddings, padding_mask

class CTBERT(lightning.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        # INPUT
        self.embedder = CTEmbeddings(config)

        # ENCODER TRANSFORMER
        self.encoder = torch.nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.hidden_size = config.hidden_size
        logger.info('saving hyperparamters')
        self.save_hyperparameters()
        
        self.loss = MaskPredictionLoss(token_weight=config.loss_tw, value_weight=config.loss_vw)
        self.optimizer_ = eval(config.optimizer)
        self.lr = config.learning_rate
        
    def forward(self, tokens, values, **kwargs):
        out = CTOutput()
        
        emb, pad_mask = self.embedder(tokens=tokens, values=values)
        emb = pad_mask.unsqueeze(-1).type_as(emb) * emb
        out.input_embeddings = emb.detach()
        
        for layer in self.encoder:
            emb = layer(
                hidden_states = emb,
                attention_mask = (-1e6 * (~pad_mask).unsqueeze(1).unsqueeze(2)).type_as(emb),
                output_attentions = kwargs.get('output_attentions', False)
            )

            if len(emb) == 2:
                emb, att = emb
            else:
                emb = emb[0]
            
            emb = pad_mask.unsqueeze(-1).type_as(emb) * emb # zeroing pad embeddings
            
            if kwargs.get('output_hidden_states', False): out.hidden_states.append(emb.detach()) 
            if kwargs.get('output_attentions', False): out.attentions.append(att.detach())

        if kwargs.get('output_last_states', False):
            out.last_hidden_state = emb.detach()
            out.last_attention = att.detach()

        # torch.matmul(out.last_hidden_state, model.embedder.token_embeddings.weight.transpose(0, 1))
        # In this implementation we are using the same weights used for the embedder to project back to the tokens (as bert implementation)
        out.token_pred = torch.matmul(emb, self.embedder.token_embeddings.weight.transpose(0, 1))
        out.value_pred = torch.matmul(emb, self.embedder.value_embeddings.weight)
        # token_pred = self.token_decoder(emb)
        # value_pred = self.value_decoder(emb)
        
        return out

    def training_step(self, batch, batch_idx):
        tokens, values, [token_labels, value_labels] = batch
        out = self.forward(
            tokens=tokens, 
            values=values
        )

        loss = self.loss(tokens, out.token_pred, out.value_pred, token_labels, value_labels)
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
        mask_values=config.dataset.mask_values
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.dataset.batch_size, 
        shuffle=config.dataset.shuffle, 
        num_workers=config.dataset.num_workers
    )

    model_config = BertConfig(
        **config.model.__dict__,
    )

    model = CTBERT(model_config)
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