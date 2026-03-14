# Step 3: Training

This step launches the mBERT pretraining loop using PyTorch Lightning.

## The Training Script

```python
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import yaml
import torch
from transformers.models.bert.modeling_bert import BertConfig
from clinical_transformer.mbert.dataset import (
    MaskedTokenDatasetFromAnnData as MaskedTokenDataset,
    collate_variable_length,
)
from clinical_transformer.mbert.modeling import LightningTrainerModel
from clinical_transformer.mbert.config import Config

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader


def pipeline():
    config_file = sys.argv[1]
    config_ = yaml.safe_load(open(config_file, "r"))
    config = Config(config_)

    # Global settings
    seed_everything(config.experiment.seed, workers=True)
    torch.set_float32_matmul_precision(
        config.experiment.set_float32_matmul_precision
    )
    config.experiment.output_dir = (
        f"{config.experiment.save_dir}/"
        f"{config.experiment.name}/"
        f"version_{config.experiment.version}/"
    )
    os.makedirs(config.experiment.output_dir, exist_ok=True)

    yaml.dump(
        config_,
        open(
            f"{config.experiment.output_dir}/experiment_{config_file.split('/')[-1]}",
            "w",
        ),
    )

    # Logger
    csv_logger = CSVLogger(
        save_dir=config.experiment.save_dir,
        name=config.experiment.name,
        version=config.experiment.version,
    )

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.experiment.output_dir}/models/",
        filename="{epoch}",
        monitor="epoch",
        save_top_k=-1,
        save_weights_only=False,
        every_n_epochs=config.trainer.save_every_n_epochs,
    )

    # Dataset and dataloader
    train_dataset = MaskedTokenDataset(
        anndata_path=config.dataset.input_file,
        context_window=config.dataset.context_window,
        mask_prob=config.dataset.masking_fraction,
        filter_zeros=config.dataset.filter_zeros,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=config.dataset.shuffle,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        persistent_workers=config.dataset.num_workers > 0,
        prefetch_factor=2 if config.dataset.num_workers > 0 else None,
        collate_fn=collate_variable_length,
    )

    # Model
    model_kwargs = config.model.__dict__.copy()
    model_kwargs.pop('enable_flash_attention', None)
    model_config = BertConfig(**model_kwargs)

    model = LightningTrainerModel(model_config)
    model.model = torch.compile(model.model)
    model_config.to_json_file(
        f"{config.experiment.output_dir}/model_config.json"
    )

    # Strategy
    if config.trainer.strategy["name"] == "deepspeed":
        strategy = DeepSpeedStrategy(
            **config.trainer.strategy.params.__dict__
        )
    else:
        strategy = config.trainer.strategy["name"]

    # Trainer
    trainer = Trainer(
        log_every_n_steps=config.trainer.log_every_n_steps,
        deterministic=config.trainer.deterministic,
        devices=config.trainer.devices,
        accelerator=config.trainer.accelerator,
        strategy=strategy,
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
        ckpt_path=config.trainer.from_checkpoint,
    )


if __name__ == "__main__":
    pipeline()
```

## Launch Command

```bash
# Multi-GPU with DDP
torchrun --nproc_per_node=8 train.py config.yaml

# Single GPU
python train.py config.yaml
```

:::{important}
Make sure `trainer.devices` in `config.yaml` matches `--nproc_per_node`. If they differ, training will fail or behave unexpectedly.
:::

## What Happens During Training

1. The dataset loads data and, for each sample:
   - randomly selects `context_window` features (if set)
   - masks the last `masking_fraction` of those features
2. `collate_variable_length` pads the batch to the max sequence length
3. The model:
   - computes embeddings
   - if FA2 backend: unpacks padded tensors → runs FA2 varlen attention → repacks
   - if SDPA backend: runs SDPA attention with appropriate mask
   - runs the prediction head **only on masked positions**
   - computes MSE loss on masked positions
4. Checkpoints are saved every `save_every_n_epochs` epochs

## Monitoring Training

Training metrics are logged as CSV files. Plot the loss curve:

```python
import pandas as pd
metrics = pd.read_csv('<save_dir>/<name>/version_<version>/metrics.csv')
metrics.plot(x='step', y='train_loss')
```

## Resuming from a Checkpoint

Set `from_checkpoint` in `config.yaml`:

```yaml
trainer:
  from_checkpoint: /path/to/models/epoch=500.ckpt
```

Then re-run the training script. Training resumes from the exact state.
