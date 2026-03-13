import sys
import os
import yaml 

import lightning
import torch
from transformers.models.bert.modeling_bert import BertLayer, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput

from clinical_transformer.pt.losses.masked_prediction import MaskPredictionLoss
from clinical_transformer.pt.training import Config

import pickle
from clinical_transformer.pt.training.rnaBERT.dataset import MaskedTokenDataset as TabularMaskedDataset
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

from typing import Optional, Tuple, Union
from dataclasses import dataclass

logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass
class nBERTModelOutput(ModelOutput):
    """
    Output class for nBERT model following HuggingFace Transformers pattern.
    
    Args:
        last_hidden_state (torch.FloatTensor): Sequence of hidden-states at the output of the last layer.
            Shape: (batch_size, sequence_length, hidden_size)
        token_predictions (torch.FloatTensor): Prediction scores for token reconstruction.
            Shape: (batch_size, sequence_length, vocab_size)
        value_predictions (torch.FloatTensor, optional): Prediction scores for value reconstruction.
            Shape: (batch_size, sequence_length)
        hidden_states (tuple, optional): Hidden-states of the model at the output of each layer.
            Tuple of torch.FloatTensor, one for each layer.
        attentions (tuple, optional): Attention weights after the attention softmax.
            Tuple of torch.FloatTensor, one for each layer.
        input_embeddings (torch.FloatTensor, optional): Input embeddings before passing through encoder.
            Shape: (batch_size, sequence_length, hidden_size)
    """
    last_hidden_state: torch.FloatTensor = None
    token_predictions: torch.FloatTensor = None
    value_predictions: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    input_embeddings: Optional[torch.FloatTensor] = None


class CTEmbeddings(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.value_embeddings = torch.nn.Linear(in_features=1, out_features=config.hidden_size, bias=True)
        self.scaling = torch.sqrt(torch.tensor(config.hidden_size))
        
        self.token_ln = torch.nn.LayerNorm(config.hidden_size)
        self.value_ln = torch.nn.LayerNorm(config.hidden_size)
        self.final_ln = torch.nn.LayerNorm(config.hidden_size)

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

        token_emb = self.token_ln(token_embeddings)
        value_emb = self.value_ln(value_embeddings)

        embeddings = (token_emb + value_emb) * self.scaling
        embeddings = self.final_ln(embeddings)

        return embeddings, padding_mask

class nBERTPretrainedModel(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # INPUT
        self.embedder = CTEmbeddings(config)

        # ENCODER TRANSFORMER
        self.encoder = torch.nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.hidden_size = config.hidden_size
        self.output_ln = torch.nn.LayerNorm(config.hidden_size)
        
        self.post_init()


    def forward(self, tokens, values, **kwargs):
        """
        Forward pass through the nBERT model.
        
        Args:
            tokens (torch.LongTensor): Input token ids. Shape: (batch_size, sequence_length)
            values (torch.FloatTensor): Input values corresponding to tokens. Shape: (batch_size, sequence_length)
            output_hidden_states (bool, optional): Whether to return hidden states for all layers.
            output_attentions (bool, optional): Whether to return attention weights for all layers.
            output_predictions (bool, optional): Whether to return token and value predictions.
            return_dict (bool, optional): Whether to return a ModelOutput instead of tuple.
            output_last_hidden_state (bool, optional): Whether to return the last hidden state.

        Returns:
            nBERTModelOutput or tuple: Model outputs including predictions and optional hidden states/attentions.
        """
        return_dict = kwargs.get('return_dict', True)
        output_hidden_states = kwargs.get('output_hidden_states', False)
        output_attentions = kwargs.get('output_attentions', False)
        output_predictions = kwargs.get('output_predictions', True)
        output_last_hidden_state = kwargs.get('output_last_hidden_state', False)
        
        # Initialize storage for optional outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Get embeddings and padding mask
        embeddings, padding_mask = self.embedder(tokens=tokens, values=values)
        embeddings = padding_mask.unsqueeze(-1).type_as(embeddings) * embeddings
        
        # Store input embeddings if requested
        input_embeddings = embeddings.detach() if kwargs.get('output_input_embeddings', False) else None
        
        # Pass through transformer layers
        hidden_state = embeddings
        for layer in self.encoder:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            
            layer_outputs = layer(
                hidden_states=hidden_state,
                attention_mask=(-1e6 * (~padding_mask).unsqueeze(1).unsqueeze(2)).type_as(hidden_state),
                output_attentions=output_attentions
            )

            if output_attentions:
                hidden_state, attention_weights = layer_outputs
                all_attentions = all_attentions + (attention_weights,)
            else:
                hidden_state = layer_outputs[0]
            
            # Zero out padding positions
            hidden_state = padding_mask.unsqueeze(-1).type_as(hidden_state) * hidden_state

        # Final layer normalization
        hidden_state = self.output_ln(hidden_state)
        hidden_state = padding_mask.unsqueeze(-1).type_as(hidden_state) * hidden_state
        
        # Add final hidden state if collecting all states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        # Generate token predictions
        token_predictions = torch.matmul(hidden_state, self.embedder.token_embeddings.weight.transpose(0, 1))
        
        # Generate value predictions (if needed in future)
        value_predictions = None  # Can be implemented later if needed
        
        # For backward compatibility, handle legacy kwargs
        if kwargs.get('output_last_states', False):
            output_last_hidden_state = True
        
        if not return_dict:
            outputs = (hidden_state, token_predictions)
            if value_predictions is not None:
                outputs = outputs + (value_predictions,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs

        return nBERTModelOutput(
            last_hidden_state=hidden_state if output_last_hidden_state else None,
            token_predictions=token_predictions if output_predictions else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            input_embeddings=input_embeddings,
        )


class LightningTrainerModel(lightning.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        # INPUT
        self.model = nBERTPretrainedModel(config)
        
        logger.info('saving hyperparamters')
        self.save_hyperparameters()
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimizer_ = eval(config.optimizer)
        self.lr = config.learning_rate
         
        
    def forward(self, **kwargs):
        out = self.model(**kwargs)

        return out

    def training_step(self, batch, batch_idx):
        tokens = batch['tokens']  # Ensure Long dtype
        token_labels = batch['original_tokens']
        values = batch['values']  # Ensure Float dtype

        out = self.forward(
            tokens=tokens,
            values=values,
            output_last_hidden_state=True,
            return_dict=True
        )

        # Create mask for masked tokens only (where input tokens == 1)
        masked_positions = (tokens == 1)
        
        # Only compute loss for masked positions to avoid unnecessary computation
        tpred = out.token_predictions[masked_positions]  # Shape: [num_masked, vocab_size]
        ttrue = token_labels[masked_positions]  # Shape: [num_masked]
        
        # Calculate loss only for masked tokens
        loss = self.loss(tpred, ttrue).mean()
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
        tokens=X_train_processed['input_ids'], 
        values=X_train_processed['gene_values'], 
        context_window=config.dataset.context_window, 
        masking_fraction=config.dataset.masking_fraction,
        mask_values=config.dataset.mask_values,
        return_cls=True,
        return_values=True
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