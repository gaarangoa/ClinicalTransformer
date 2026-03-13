# Step 3: Training

This step launches the vnBERT pretraining loop using PyTorch Lightning and DeepSpeed.

## The Training Script

The `train.py` script reads the config, builds the dataset and model, and runs the training loop. Below is the complete script with comments:

```python
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import yaml
import torch
from transformers.models.bert.modeling_bert import BertConfig
from clinical_transformer.vnbert.dataset import (
    MaskedTokenDatasetFromAnnData as MaskedTokenDataset,
)
from clinical_transformer.vnbert.modeling import LightningTrainerModel
from clinical_transformer.utils.config import Config

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
    torch.backends.cuda.enable_flash_sdp(
        enabled=config.model.enable_flash_attention
    )
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

    # Save a copy of the config to the output directory
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
    )

    # Model
    model_config = BertConfig(**config.model.__dict__)
    model = LightningTrainerModel(model_config)
    model_config.to_json_file(
        f"{config.experiment.output_dir}/model_config.json"
    )

    # Strategy
    if config.trainer.strategy["name"] == "deepspeed":
        ds_strategy = DeepSpeedStrategy(
            **config.trainer.strategy.params.__dict__
        )
    else:
        ds_strategy = config.trainer.strategy["name"]

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
        ckpt_path=config.trainer.from_checkpoint,
    )


if __name__ == "__main__":
    pipeline()
```

## Launch Command

Create a `train.sh` file or run directly from the terminal:

```bash
# Multi-GPU: set --nproc_per_node to match trainer.devices in config.yaml
torchrun --nproc_per_node=8 train.py config.yaml
```

For a single GPU:

```bash
torchrun --nproc_per_node=1 train.py config.yaml
```

:::{important}
Make sure `trainer.devices` in `config.yaml` matches `--nproc_per_node`. If they differ, the training will fail or behave unexpectedly.
:::

## What Happens During Training

1. The `MaskedTokenDataset` loads the `.h5ad` file and, for each sample:
   - randomly selects `context_window` features
   - masks `masking_fraction` of those features
2. The `LightningTrainerModel` processes each batch through the BERT encoder
3. The model predicts the **values** of masked features using MSE loss
4. Checkpoints are saved every `save_every_n_epochs` epochs to `<output_dir>/models/`

## Monitoring Training

Training metrics are logged as CSV files in:

```
<save_dir>/<name>/version_<version>/
```

Plot the loss curve:

```python
import pandas as pd
metrics = pd.read_csv('<save_dir>/<name>/version_<version>/metrics.csv')
metrics.plot(x='step', y='train_loss')
```

## Resuming from a Checkpoint

If training is interrupted, set `from_checkpoint` in `config.yaml`:

```yaml
trainer:
  from_checkpoint: /path/to/models/My_FM_NAME/version_1/models/epoch=500.ckpt
```

Then re-run `train.sh`. Training resumes from the exact state (model weights, optimizer, epoch counter).
