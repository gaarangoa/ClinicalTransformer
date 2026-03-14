# Step 2: Configuration

All training parameters are controlled by a single `config.yaml` file. This page explains every field.

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
  input_file: /dev/shm/training_data.h5ad  # .h5ad or .pkl file from Step 1
  priors_file:                              # Optional, leave empty if unused
  num_workers: 23           # Data-loading workers (num_cpus - 1)
  batch_size: 1184          # Samples per batch per GPU
  context_window: 1000      # Features sampled per training step
  masking_fraction: 0.3     # Fraction of features to mask
  filter_zeros: False       # Skip zero-valued features
  shuffle: True

model:
  modalities: 1             # Number of data modalities
  vocab_size: 30            # Must be >= tokenizer.vocab_size
  hidden_size: 256          # Embedding dimension
  intermediate_size: 1024   # Feed-forward hidden size (typically 4 × hidden_size)
  num_attention_heads: 4    # Attention heads
  num_hidden_layers: 4      # Transformer layers
  use_scgpt_mask: false     # true: scGPT attention masking, false: full bidirectional
  attention_backend: 'sdpa' # 'sdpa' (default) or 'fa2' (flash_attn varlen)
  enable_flash_attention: true  # Legacy flag, ignored when using custom SDPA layers
  dropout: 0.1
  learning_rate: 1.0e-5
  optimizer: 'torch.optim.AdamW'  # 'torch.optim.Adam|torch.optim.AdamW|FusedAdam|DeepSpeedCPUAdam'
  pad_token_id: 0

trainer:
  epochs: 1000
  log_every_n_steps: 25
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
    name: ddp               # 'ddp' for data parallelism, 'deepspeed' for ZeRO sharding
    params:
      stage: 2              # Only used with deepspeed
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
| `input_file` | Path to the `.h5ad` or `.pkl` file created in {doc}`build-dataset`. |
| `num_workers` | Number of CPU processes for data loading. Set to `num_cpus - 1`. |
| `batch_size` | Samples per batch **per GPU**. Reduce if you get OOM errors. |
| `context_window` | How many features the model sees per training step. Set to `null` for variable-length (requires `collate_variable_length`). |
| `masking_fraction` | Fraction of context-window features to mask. 0.15--0.30 is typical. |
| `filter_zeros` | If `True`, zero-valued features are excluded from sampling. |

### model

| Field | Description |
|-------|-------------|
| `vocab_size` | Must be >= `tokenizer.vocab_size` from Step 1. |
| `hidden_size` | Embedding dimension. Must be divisible by `num_attention_heads`. |
| `intermediate_size` | Feed-forward network hidden size. Typically `4 × hidden_size`. |
| `num_attention_heads` | Number of attention heads per layer. |
| `num_hidden_layers` | Depth of the transformer encoder. |
| `use_scgpt_mask` | `true`: scGPT attention masking (blocks masked-to-masked attention). `false`: full bidirectional (recommended for speed). |
| `attention_backend` | `'sdpa'`: PyTorch native SDPA (works everywhere). `'fa2'`: Flash Attention 2 varlen (Ampere/Hopper GPUs, requires `pip install flash-attn`). |
| `dropout` | Dropout rate applied in attention and feed-forward layers. |
| `learning_rate` | Initial learning rate. |
| `optimizer` | `torch.optim.AdamW` (recommended), `FusedAdam` (with DeepSpeed), or `DeepSpeedCPUAdam` (CPU-offloaded). |

### trainer

| Field | Description |
|-------|-------------|
| `epochs` | Total training epochs. |
| `devices` | Number of GPUs to use. |
| `precision` | `bf16-mixed` (recommended), `16-mixed`, or `32`. |
| `log_every_n_steps` | How often to log metrics. 25-50 is reasonable; 1 adds overhead. |
| `num_nodes` | Number of compute nodes for multi-node training. |
| `save_every_n_epochs` | Checkpoint save frequency. |
| `accumulate_grad_batches` | Accumulate gradients over N batches before updating. |
| `from_checkpoint` | Path to a checkpoint to resume training from. Leave empty to start fresh. |

### trainer.strategy

| Field | Description |
|-------|-------------|
| `name` | `ddp` (recommended when model fits in GPU memory) or `deepspeed` (for ZeRO sharding). |
| `stage` | DeepSpeed ZeRO stage. **Stage 1**: shards optimizer states. **Stage 2**: also shards gradients. **Stage 3**: also shards parameters. |
| `offload_optimizer` | Move optimizer states to CPU RAM. Slower, but saves GPU memory. |
| `offload_parameters` | Move model parameters to CPU RAM (stage 3 only). |

## Hardware Recommendations

| Hardware | Strategy | attention_backend | use_scgpt_mask | Notes |
|----------|----------|-------------------|----------------|-------|
| 1× GPU (any) | `auto` | `sdpa` | `false` | No multi-GPU overhead |
| 8× A10 (24GB) | `ddp` | `fa2` | `false` | FA2 supported on Ampere; model fits in 24GB |
| 8× A100 (80GB) | `ddp` | `fa2` | `false` | Largest batches; FA2 native |
| 8× H100 | `ddp` | `fa2` | `false` | Fastest; FA2 + Hopper optimizations |
| 1× Blackwell (GB10) | `auto` | `sdpa` | `false` | FA2 not yet supported; SDPA flash works |
| Limited GPU memory | `deepspeed` (stage 2) | `sdpa` | either | Shard optimizer + gradients |

## Model Sizing Guide

| Size | hidden_size | intermediate_size | num_attention_heads | num_hidden_layers | ~Params |
|------|-------------|-------------------|---------------------|-------------------|---------|
| Tiny | 128 | 512 | 2 | 2 | ~2M |
| Small | 256 | 1024 | 4 | 4 | ~10M |
| Medium | 512 | 2048 | 8 | 8 | ~40M |
| Large | 1024 | 4096 | 16 | 12 | ~96M |
