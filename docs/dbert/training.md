# Step 3: Training

This step launches the dBERT pretraining loop using PyTorch Lightning and DeepSpeed.

## The Training Script

```python
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import yaml
import torch
from transformers.models.bert.modeling_bert import BertConfig
from clinical_transformer.dbert.dataset import (
    MaskedTokenDatasetFromAnnData as MaskedTokenDataset,
)
from clinical_transformer.dbert.modeling import LightningTrainerModel
from clinical_transformer.dbert.config import Config

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

    csv_logger = CSVLogger(
        save_dir=config.experiment.save_dir,
        name=config.experiment.name,
        version=config.experiment.version,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.experiment.output_dir}/models/",
        filename="{epoch}",
        monitor="epoch",
        save_top_k=-1,
        save_weights_only=False,
        every_n_epochs=config.trainer.save_every_n_epochs,
    )

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

    model_config = BertConfig(**config.model.__dict__)
    model = LightningTrainerModel(model_config)
    model_config.to_json_file(
        f"{config.experiment.output_dir}/model_config.json"
    )

    if config.trainer.strategy["name"] == "deepspeed":
        ds_strategy = DeepSpeedStrategy(
            **config.trainer.strategy.params.__dict__
        )
    else:
        ds_strategy = config.trainer.strategy["name"]

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

```bash
# Multi-GPU
torchrun --nproc_per_node=8 train.py config.yaml

# Single GPU
torchrun --nproc_per_node=1 train.py config.yaml
```

:::{important}
Make sure `trainer.devices` in `config.yaml` matches `--nproc_per_node`.
:::

## What Happens During Training

1. The `MaskedTokenDataset` loads the `.h5ad` file and for each sample:
   - randomly selects `context_window` features
   - masks `masking_fraction` of those features (values set to `-10.0`)
2. The `LightningTrainerModel` processes each batch through the dBERT encoder
3. The disentangled attention mask splits masked positions into isolated and contextual groups based on `mask1_ratio`
4. The model predicts the **values** of all masked features using MSE loss
5. If `mask1_warmup_epochs > 0`, the `mask1_ratio` is annealed from 0 to target at the start of each epoch
6. Checkpoints are saved every `save_every_n_epochs` epochs

## Monitoring Training

```python
import pandas as pd
metrics = pd.read_csv('<save_dir>/<name>/version_<version>/metrics.csv')
metrics.plot(x='step', y='train_loss')
```

## Resuming from a Checkpoint

Set `from_checkpoint` in `config.yaml`:

```yaml
trainer:
  from_checkpoint: /path/to/models/My_FM_NAME/version_1/models/epoch=500.ckpt
```

Then re-run `train.sh`.
