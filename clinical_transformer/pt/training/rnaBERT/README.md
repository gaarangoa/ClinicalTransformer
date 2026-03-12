# RNA-BERT: Gene Expression Transformer Model

This repository contains the implementation of RNA-BERT, a transformer-based model designed specifically for gene expression data analysis using rank-based normalization and masked language modeling.

## Overview

RNA-BERT adapts the BERT architecture for gene expression data by:
- Using rank-based normalization instead of traditional min-max scaling
- Combining token embeddings (gene identifiers) with value embeddings (expression levels)
- Implementing masked gene prediction for self-supervised learning

## Architecture

### Core Components

#### 1. CTEmbeddings Module
The embedding layer combines two types of information:
- **Token Embeddings**: Maps gene identifiers to dense vectors
- **Value Embeddings**: Projects normalized expression values to the same embedding space

```python
embeddings = (token_embeddings + value_embeddings) * scaling_factor
```

**Key Features:**
- Separate layer normalization for tokens and values
- Scaling factor based on hidden dimension size
- Final layer normalization for stability

#### 2. nBERTPretrainedModel
The main transformer model inheriting from `BertPreTrainedModel`:
- **Encoder**: Stack of BERT layers for contextualized representations
- **Output Layer**: Projects hidden states back to vocabulary space for prediction
- **Attention Masking**: Handles padding tokens appropriately

### Model Forward Pass

1. **Embedding**: Combines token and value embeddings
2. **Encoding**: Passes through transformer layers with attention masking
3. **Prediction**: Projects final hidden states to vocabulary logits
4. **Masking**: Applies padding mask to zero out irrelevant positions

## Data Processing Pipeline

### Rank-Based Normalization
Unlike traditional approaches, RNA-BERT uses rank-based normalization:

1. **Ranking**: Sort genes by expression level within each sample
2. **Normalization**: Divide ranks by maximum rank (0-1 scale)
3. **Ordering**: Maintain descending order by original expression values

This approach makes the model robust to:
- Different expression scales across experiments
- Outliers in expression data
- Technical variations between platforms

### Dataset Structure
The `MaskedTokenDataset` handles:
- **Context Windowing**: Random sampling of gene subsequences
- **Masking**: Random masking of ~15% of genes for self-supervised learning
- **Padding**: Ensures uniform sequence lengths
- **CLS Token**: Optional classification token for downstream tasks

## Training Process

### Lightning Module Architecture
The `LightningTrainerModel` implements:

#### Loss Function
Optimized masked language modeling loss:
```python
# Only compute loss for masked positions (tokens == 1)
masked_positions = (tokens == 1)
predictions = model_output[masked_positions]
targets = labels[masked_positions]
loss = CrossEntropyLoss(predictions, targets)
```

**Benefits:**
- Computational efficiency (only ~15% of tokens)
- Memory optimization
- Faster training convergence

#### Optimizer Configuration
Supports various optimizers through configuration:
- DeepSpeed optimizers for large-scale training
- Standard PyTorch optimizers
- Configurable learning rates and schedules

### Training Pipeline

#### 1. Configuration Loading
```python
config = Config(yaml.safe_load(config_file))
```

#### 2. Data Preparation
```python
# Load preprocessed data
X_train = pickle.load(config.dataset.input_file)
dataset = MaskedTokenDataset(
    tokens=X_train['input_ids'],
    values=X_train['gene_values'],
    context_window=512,
    masking_fraction=0.15
)
```

#### 3. Model Initialization
```python
model_config = BertConfig(**config.model.__dict__)
model = LightningTrainerModel(model_config)
```

#### 4. Training Setup
- **Logging**: CSV logger for metrics tracking
- **Checkpointing**: Model saving at specified intervals
- **Distributed Training**: DeepSpeed strategy support
- **Mixed Precision**: Configurable precision training

## Key Features

### 1. Rank-Based Gene Expression Processing
- Converts absolute expression values to relative rankings
- Robust to batch effects and technical variations
- Preserves biological signal while normalizing scale

### 2. Efficient Masked Language Modeling
- Only computes loss for masked tokens
- Significant speedup over naive implementations
- Memory-efficient training on long sequences

### 3. Flexible Architecture
- Configurable model dimensions
- Scalable to different vocabulary sizes
- Support for various downstream tasks

### 4. Production-Ready Training
- Lightning framework for robust training loops
- DeepSpeed integration for large models
- Comprehensive logging and checkpointing

## Configuration

### Model Parameters
```yaml
model:
  vocab_size: 20000        # Number of unique genes
  hidden_size: 768         # Embedding dimension
  num_hidden_layers: 12    # Transformer layers
  num_attention_heads: 12  # Attention heads per layer
  intermediate_size: 3072  # Feed-forward dimension
  pad_token_id: 0         # Padding token
```

### Training Parameters
```yaml
dataset:
  context_window: 512      # Maximum sequence length
  masking_fraction: 0.15   # Fraction of tokens to mask
  batch_size: 32          # Training batch size

trainer:
  max_epochs: 100         # Training epochs
  learning_rate: 1e-4     # Initial learning rate
  precision: 16           # Mixed precision training
```

## Usage

### Basic Training
```bash
python model.py config.yaml
```

### Configuration File Example
```yaml
experiment:
  name: "rna_bert_experiment"
  save_dir: "./experiments"
  seed: 42

model:
  vocab_size: 20000
  hidden_size: 768
  num_hidden_layers: 12

dataset:
  input_file: "preprocessed_data.pkl"
  context_window: 512
  masking_fraction: 0.15
```

## Input Data Format

The model expects preprocessed data with:
```python
{
    'input_ids': List[List[int]],    # Token sequences
    'gene_values': List[List[float]] # Normalized expression values
}
```

Where:
- `input_ids`: Gene token identifiers
- `gene_values`: Rank-normalized expression values (0-1 range)

## Model Outputs

During training, the model produces:
- **Token Predictions**: Logits over vocabulary for masked positions
- **Hidden States**: Contextualized gene representations
- **Attention Weights**: Gene-gene attention patterns (optional)

## Key Innovations

1. **Gene Expression Tokenization**: Novel approach to representing continuous expression as discrete tokens with associated values

2. **Rank Normalization**: Robust normalization scheme that preserves relative gene importance

3. **Efficient Masking**: Optimized loss computation for masked language modeling

4. **Biologically-Informed Architecture**: Design choices motivated by gene expression characteristics

## Dependencies

- PyTorch Lightning
- Transformers (Hugging Face)
- DeepSpeed (optional, for large-scale training)
- PyTorch
- NumPy
- YAML

## Future Extensions

Potential improvements and extensions:
- Multi-task learning with clinical outcomes
- Cross-species transfer learning
- Integration with other omics data types
- Interpretability tools for biological insights

---

This implementation provides a foundation for transformer-based analysis of gene expression data, combining the power of modern NLP architectures with domain-specific biological considerations.
