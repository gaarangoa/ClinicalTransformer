# Step 2: Configuration

All training parameters are controlled by a single `config.yaml` file. This page explains every field so you can adapt it to any dataset.

## Full Annotated Template

```yaml
experiment:
  name: My_FM_NAME                     # Experiment name (used in output paths)
  version: 1                           # Version number
  save_dir: /path/to/models/ssl/       # Root output directory
  seed: 0                              # Random seed
  set_float32_matmul_precision: medium # Options: highest | high | medium
  comments: ""

dataset:
  input_file: /dev/shm/training_data.h5ad  # .h5ad file from Step 1
  priors_file:                              # Optional, leave empty if unused
  num_workers: 23           # Data-loading workers (num_cpus - 1)
  batch_size: 1184          # Samples per batch per GPU
  context_window: 22        # Features sampled per training step
  masking_fraction: 0.3     # Fraction of features to mask
  filter_zeros: False       # Skip zero-valued features
  shuffle: True

model:
  modalities: 1             # Number of data modalities
  vocab_size: 30            # Must be >= tokenizer.vocab_size
  hidden_size: 128          # Embedding dimension
  intermediate_size: 512    # Feed-forward hidden size
  num_attention_heads: 2    # Attention heads
  num_hidden_layers: 2      # Transformer layers
  enable_flash_attention: true
  dropout: 0.1
  _attn_implementation: 'sdpa'
  learning_rate: 1.0e-5
  optimizer: 'torch.optim.Adam'
  pad_token_id: 0

trainer:
  epochs: 1000
  log_every_n_steps: 1
  deterministic: false
  devices: 8                # Number of GPUs
  accelerator: 'cuda'
  precision: 'bf16-mixed'
  num_nodes: 1
  reload_dataloaders_every_n_epochs: False
  save_every_n_epochs: 100
  accumulate_grad_batches: 1
  from_checkpoint:          # Leave empty to start fresh
  strategy:
    name: deepspeed
    params:
      stage: 2
      offload_optimizer: False
      offload_parameters: False
```

## Field Reference

### experiment

| Field | Description |
|-------|-------------|
| `name` | Used to create the output directory: `<save_dir>/<name>/version_<version>/` |
| `version` | Integer version number. Increment to keep previous runs. |
| `save_dir` | Root directory where all outputs are written. |
| `seed` | Random seed for reproducibility. |
| `set_float32_matmul_precision` | Controls CUDA matmul precision. `medium` is a good balance of speed and accuracy. |

### dataset

| Field | Description |
|-------|-------------|
| `input_file` | Path to the `.h5ad` file created in {doc}`build-dataset`. |
| `num_workers` | Number of CPU processes for data loading. Set to `num_cpus - 1`. |
| `batch_size` | Samples per batch **per GPU**. Reduce if you get OOM errors. |
| `context_window` | How many features the model sees per training step. Set equal to total feature count for small datasets (<100 features), or use a subset for large vocabularies (e.g. 1000 genes out of 20000). |
| `masking_fraction` | Fraction of context-window features to mask. 0.15&ndash;0.30 is typical. |
| `filter_zeros` | If `True`, zero-valued features are excluded from sampling. |

### model

| Field | Description |
|-------|-------------|
| `vocab_size` | Must be &ge; `tokenizer.vocab_size` from Step 1. |
| `hidden_size` | Embedding dimension. Must be divisible by `num_attention_heads`. |
| `intermediate_size` | Feed-forward network hidden size. Typically `4 &times; hidden_size`. |
| `num_attention_heads` | Number of attention heads per layer. |
| `num_hidden_layers` | Depth of the transformer encoder. |
| `enable_flash_attention` | Use Flash/SDPA attention for speed and memory savings. |
| `dropout` | Dropout rate applied in attention and feed-forward layers. |
| `learning_rate` | Initial learning rate. |
| `optimizer` | `torch.optim.Adam` or `DeepSpeedCPUAdam` (for CPU-offloaded training). |

### trainer

| Field | Description |
|-------|-------------|
| `epochs` | Total training epochs. |
| `devices` | Number of GPUs to use. |
| `precision` | `bf16-mixed` (recommended), `16-mixed`, or `32`. |
| `num_nodes` | Number of compute nodes for multi-node training. |
| `save_every_n_epochs` | Checkpoint save frequency. |
| `accumulate_grad_batches` | Accumulate gradients over N batches before updating. Increase if mixed-precision training is unstable. |
| `from_checkpoint` | Path to a checkpoint to resume training from. Leave empty to start fresh. |

### trainer.strategy

| Field | Description |
|-------|-------------|
| `name` | `deepspeed` (recommended) or any PyTorch Lightning strategy name. |
| `stage` | DeepSpeed ZeRO stage. **Stage 2**: shards optimizer states and gradients. **Stage 3**: also shards model parameters (use when the model doesn't fit in GPU memory). |
| `offload_optimizer` | Move optimizer states to CPU RAM. Slower, but saves GPU memory. |
| `offload_parameters` | Move model parameters to CPU RAM. Even slower, but enables very large models. |

## Model Sizing Guide

| Size | hidden_size | intermediate_size | num_attention_heads | num_hidden_layers | ~Params |
|------|-------------|-------------------|---------------------|-------------------|---------|
| Tiny | 128 | 512 | 2 | 2 | ~2M |
| Small | 256 | 1024 | 4 | 4 | ~10M |
| Medium | 512 | 2048 | 8 | 8 | ~40M |
| Large | 1024 | 4096 | 16 | 12 | ~96M |

## DeepSpeed Strategy Guide

| Scenario | stage | offload_optimizer | offload_parameters |
|----------|-------|-------------------|--------------------|
| Model fits in GPU memory | 2 | False | False |
| Tight on GPU memory | 2 | True | False |
| Model doesn't fit in GPU memory | 3 | True | False |
| Very large model, limited GPU memory | 3 | True | True |
