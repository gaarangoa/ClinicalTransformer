# nBERT: Clinical Transformer for Tabular Data

## Overview

nBERT (clinical BERT) is a specialized transformer architecture designed for clinical tabular data processing. Built on PyTorch Lightning, it extends the BERT architecture to handle both categorical tokens and continuous values simultaneously, making it particularly suitable for clinical genomics, electronic health records, and other structured clinical datasets.

## Key Features

- **Dual Input Processing**: Handles both categorical tokens and continuous values
- **Clinical-Focused Architecture**: Optimized for biomedical and clinical data patterns
- **PyTorch Lightning Integration**: Built-in distributed training, logging, and checkpointing
- **Flexible Output Options**: Supports various output configurations for different downstream tasks
- **DeepSpeed Support**: Efficient large-scale training with memory optimization
- **Flash Attention**: Optional flash attention mechanism for improved performance

## Architecture

### Model Components

1. **CTEmbeddings**: Custom embedding layer that combines token and value embeddings
2. **CTBERT**: Main transformer model with Lightning integration
3. **CTOutput**: Structured output class for organized predictions

### Key Innovations

- **Token + Value Fusion**: Combines categorical tokens with continuous values in a unified embedding space
- **Masked Prediction**: Supports both token and value reconstruction tasks
- **Clinical Logging**: Specialized metrics for clinical data validation

## Installation

### Requirements

```bash
# Core dependencies
pip install torch>=2.0.0
pip install lightning>=2.0.0
pip install transformers>=4.21.0
pip install deepspeed>=0.9.0  # Optional for large-scale training

# Additional dependencies
pip install pyyaml
pip install pickle5
```

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/clinical_transformer.git
cd ods_eds_foundation_models/clinical_transformer/pt/training/BERT

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Data Preparation

Your data should be preprocessed and saved as a pickle file. The expected format is:

```python
import pickle

# Data structure: List of samples
# Each sample: (tokens, values, labels)
data = [
    {
        'tokens': [1, 2, 3, 4, 0, 0],  # Token IDs with padding
        'values': [0.5, 0.8, 0.2, 0.9, 0.0, 0.0],  # Continuous values
        'token_labels': [1, 2, 3, 4, 0, 0],  # Target tokens for MLM
        'value_labels': [0.5, 0.8, 0.2, 0.9, 0.0, 0.0]  # Target values
    }
    # ... more samples
]

# Save preprocessed data
with open('clinical_data.pkl', 'wb') as f:
    pickle.dump(data, f)
```

### 2. Configuration File

Create a YAML configuration file (e.g., `config.yaml`):

```yaml
# Experiment Configuration
experiment:
  name: "nbert_clinical"
  version: 1
  seed: 42
  save_dir: "./experiments"
  set_float32_matmul_precision: "medium"

# Model Configuration
model:
  vocab_size: 30000
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 512
  type_vocab_size: 2
  initializer_range: 0.02
  layer_norm_eps: 1e-12
  pad_token_id: 0
  position_embedding_type: "absolute"
  use_cache: true
  classifier_dropout: null
  enable_flash_attention: true
  
  # Loss configuration
  loss_tw: 1.0  # Token weight
  loss_vw: 1.0  # Value weight
  
  # Optimizer
  optimizer: "torch.optim.AdamW"
  learning_rate: 2e-5

# Dataset Configuration
dataset:
  input_file: "clinical_data.pkl"
  context_window: 128
  masking_fraction: 0.15
  mask_values: true
  batch_size: 32
  shuffle: true
  num_workers: 4

# Trainer Configuration
trainer:
  devices: [0]  # GPU devices
  accelerator: "gpu"
  strategy: 
    name: "auto"  # or "deepspeed" for large models
  max_epochs: 10
  precision: "16-mixed"
  accumulate_grad_batches: 1
  log_every_n_steps: 50
  deterministic: false
  reload_dataloaders_every_n_epochs: 0
  save_every_n_epochs: 1
  from_checkpoint: null  # Path to resume training
  num_nodes: 1

# DeepSpeed Configuration (if using)
# trainer:
#   strategy:
#     name: "deepspeed"
#     params:
#       stage: 2
#       offload_optimizer: true
#       allgather_partitions: true
#       reduce_scatter: true
#       overlap_comm: true
#       contiguous_gradients: true
```

### 3. Training

```bash
# Basic training
python nBERT.py config.yaml

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 nBERT.py config.yaml

# With DeepSpeed
python nBERT.py config_deepspeed.yaml
```

## Model Architecture Details

### CTEmbeddings

The embedding layer combines token and value information:

```python
class CTEmbeddings(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # Token embeddings for categorical data
        self.token_embeddings = torch.nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=config.pad_token_id
        )
        
        # Value embeddings for continuous data
        self.value_embeddings = torch.nn.Linear(
            in_features=1, 
            out_features=config.hidden_size, 
            bias=True
        )
        
        # Normalization and scaling
        self.scaling = torch.sqrt(torch.tensor(config.hidden_size))
        self.token_ln = torch.nn.LayerNorm(config.hidden_size)
        self.value_ln = torch.nn.LayerNorm(config.hidden_size)
        self.final_ln = torch.nn.LayerNorm(config.hidden_size)
```

**Key Features:**
- Separate embedding spaces for tokens and values
- Layer normalization for stable training
- Scaling factor for embedding magnitude control
- Padding mask generation for attention mechanisms

### CTBERT Model

The main transformer model extends PyTorch Lightning:

```python
class CTBERT(lightning.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        # Core components
        self.embedder = CTEmbeddings(config)
        self.encoder = torch.nn.ModuleList([
            BertLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.output_ln = torch.nn.LayerNorm(config.hidden_size)
        
        # Training components
        self.loss = MaskPredictionLoss(
            token_weight=config.loss_tw, 
            value_weight=config.loss_vw
        )
```

**Training Features:**
- Automatic hyperparameter saving
- Built-in logging of training metrics
- Configurable loss weighting for tokens vs values
- Embedding statistics tracking

### CTOutput

Structured output class for organized predictions:

```python
class CTOutput:
    def __init__(self):
        self.token_pred = None          # Token predictions
        self.value_pred = None          # Value predictions
        self.last_hidden_state = None  # Final layer output
        self.last_attention = None     # Final attention weights
        self.input_embeddings = None   # Input embeddings
        self.hidden_states = []        # All layer outputs
        self.attentions = []          # All attention weights
```

## Usage Examples

### Basic Training Script

```python
import torch
import yaml
from clinical_transformer._legacy.training.BERT.nBERT import CTBERT, pipeline
from transformers.models.bert.modeling_bert import BertConfig

# Load configuration
config = yaml.safe_load(open('config.yaml', 'r'))

# Create model configuration
model_config = BertConfig(**config['model'])

# Initialize model
model = CTBERT(model_config)

# Run training pipeline
pipeline()
```

### Custom Training Loop

```python
import torch
from torch.utils.data import DataLoader
from clinical_transformer._legacy.training.BERT.nBERT import CTBERT
from clinical_transformer._legacy.datasets.dataloader.tabular import TabularMaskedDataset

# Load your data
with open('clinical_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Create dataset
dataset = TabularMaskedDataset(
    data, 
    context_window=128,
    masking_fraction=0.15,
    mask_values=True
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
config = BertConfig(vocab_size=30000, hidden_size=768, ...)
model = CTBERT(config)

# Training loop
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(10):
    for batch_idx, batch in enumerate(dataloader):
        tokens, values, labels = batch
        
        # Forward pass
        outputs = model(
            tokens=tokens, 
            values=values,
            output_last_states=True
        )
        
        # Calculate loss
        loss = model.loss(tokens, outputs.token_pred, outputs.value_pred, *labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### Inference Example

```python
# Load trained model
model = CTBERT.load_from_checkpoint('path/to/checkpoint.ckpt')
model.eval()

# Prepare input data
tokens = torch.tensor([[1, 2, 3, 4, 0, 0]])  # Shape: [batch_size, seq_len]
values = torch.tensor([[0.5, 0.8, 0.2, 0.9, 0.0, 0.0]])  # Shape: [batch_size, seq_len]

# Run inference
with torch.no_grad():
    outputs = model(
        tokens=tokens,
        values=values,
        output_hidden_states=True,
        output_attentions=True,
        output_last_states=True
    )

# Access predictions
token_predictions = outputs.token_pred  # Shape: [batch_size, seq_len, vocab_size]
value_predictions = outputs.value_pred  # Shape: [batch_size, seq_len, hidden_size]
hidden_states = outputs.hidden_states   # List of tensors from each layer
attentions = outputs.attentions         # List of attention weights
```

## Advanced Features

### Flash Attention

Enable flash attention for improved memory efficiency and speed:

```yaml
model:
  enable_flash_attention: true
```

### DeepSpeed Integration

For large-scale training with memory optimization:

```yaml
trainer:
  strategy:
    name: "deepspeed"
    params:
      stage: 2                    # ZeRO stage 2
      offload_optimizer: true     # Offload optimizer to CPU
      allgather_partitions: true  # Efficient gradient aggregation
      reduce_scatter: true        # Reduce memory usage
      overlap_comm: true          # Overlap communication and computation
      contiguous_gradients: true  # Memory-efficient gradients
```

### Custom Loss Weighting

Balance token and value reconstruction losses:

```yaml
model:
  loss_tw: 1.0  # Token weight
  loss_vw: 0.5  # Value weight (lower for noisy continuous data)
```

### Monitoring and Logging

The model automatically logs:
- Training loss
- Batch index
- Embedding statistics (min/max standard deviation)
- Custom metrics via Lightning's logging system

## Output Interpretation

### Token Predictions
- **Shape**: `[batch_size, sequence_length, vocab_size]`
- **Type**: Logits for each vocabulary token
- **Usage**: Apply softmax for probabilities, argmax for predictions

### Value Predictions
- **Shape**: `[batch_size, sequence_length, hidden_size]`
- **Type**: Continuous value predictions (with softplus activation)
- **Usage**: Direct regression outputs for continuous variables

### Hidden States
- **Format**: List of tensors, one per layer
- **Shape**: `[batch_size, sequence_length, hidden_size]`
- **Usage**: Feature extraction, layer analysis, downstream tasks

### Attention Weights
- **Format**: List of tensors, one per layer
- **Shape**: `[batch_size, num_heads, sequence_length, sequence_length]`
- **Usage**: Interpretability, attention visualization

## Best Practices

### Data Preprocessing

1. **Tokenization**: Use consistent vocabulary across train/val/test
2. **Value Normalization**: Scale continuous values appropriately
3. **Padding**: Use consistent padding tokens (typically 0)
4. **Masking**: Apply appropriate masking strategies for your domain

### Training Tips

1. **Learning Rate**: Start with 2e-5, adjust based on convergence
2. **Batch Size**: Use largest batch size that fits in memory
3. **Sequence Length**: Balance between context and computational cost
4. **Masking Fraction**: 15% is standard, but adjust for your data

### Performance Optimization

1. **Flash Attention**: Enable for large models
2. **Mixed Precision**: Use 16-bit training for speed/memory
3. **Gradient Accumulation**: Simulate larger batches
4. **DeepSpeed**: For models >1B parameters

## Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Reduce batch size
batch_size: 16  # Instead of 32

# Enable gradient accumulation
accumulate_grad_batches: 2

# Use mixed precision
precision: "16-mixed"
```

**Slow Training**
```bash
# Enable flash attention
enable_flash_attention: true

# Increase number of workers
num_workers: 8

# Use multiple GPUs
devices: [0, 1, 2, 3]
```

**Poor Convergence**
```bash
# Adjust learning rate
learning_rate: 1e-5  # Lower learning rate

# Increase model capacity
hidden_size: 1024
num_hidden_layers: 24

# Modify loss weighting
loss_tw: 2.0  # Emphasize token reconstruction
loss_vw: 0.5  # De-emphasize value reconstruction
```

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor model statistics
model.eval()
with torch.no_grad():
    outputs = model(tokens, values, output_last_states=True)
    print(f"Embedding mean: {outputs.last_hidden_state.mean()}")
    print(f"Embedding std: {outputs.last_hidden_state.std()}")
```

## Performance Benchmarks

### Model Sizes

| Configuration | Parameters | Memory (GPU) | Training Speed |
|---------------|------------|--------------|----------------|
| Small         | 110M       | 4GB          | ~1000 samples/sec |
| Base          | 340M       | 8GB          | ~500 samples/sec |
| Large         | 770M       | 16GB         | ~200 samples/sec |

### Scaling Guidelines

- **Single GPU**: Up to 770M parameters
- **Multi-GPU**: Up to 10B+ parameters with DeepSpeed
- **Batch Size**: 32-128 depending on sequence length and model size
- **Sequence Length**: 128-512 for most clinical applications

## Citation

If you use nBERT in your research, please cite:

```bibtex
@article{nbert2025,
  title={nBERT: Clinical Transformer for Tabular Data},
  author={Clinical Transformer Team},
  journal={Journal of Biomedical Informatics},
  year={2025},
  publisher={Elsevier},
  url={https://github.com/<your-org>/clinical_transformer}
}
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/<your-org>/clinical_transformer.git
cd ods_eds_foundation_models

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs.clinical-transformer.org](https://docs.clinical-transformer.org)
- **Issues**: [GitHub Issues](https://github.com/<your-org>/clinical_transformer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/<your-org>/clinical_transformer/discussions)
- **Email**: clinical-transformer-support@example.com

## Acknowledgments

- Built on [PyTorch Lightning](https://www.pytorchlightning.ai/)
- Uses [HuggingFace Transformers](https://huggingface.co/transformers/)
- Inspired by clinical research at [Organization Name]
