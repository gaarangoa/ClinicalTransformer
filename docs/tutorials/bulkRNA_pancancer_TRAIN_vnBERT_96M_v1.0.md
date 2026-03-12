
# Training vnBERT 96M Foundation Model on Bulk RNA-seq Data

## Overview

This tutorial demonstrates how to train a vnBERT (Value-based BERT) foundation model using the Clinical Transformer framework. vnBERT is a specialized transformer architecture designed for data that leverages a value-based encoding approach to handle numerical gene expression values from RNA sequencing data or any other modality.

## Background

### vnBERT Architecture
vnBERT (Value-based BERT) is a transformer-based foundation model specifically designed for omics data. Unlike traditional BERT models that work with discrete tokens, vnBERT can directly process continuous numerical values representing e.g., gene expression levels. This makes it particularly well-suited for genomic applications where preserving the quantitative nature of gene expression is crucial.

### Dataset
We use RNA-seq data from two major cancer genomics consortiums:
- **TCGA (The Cancer Genome Atlas)**: A comprehensive cancer genomics dataset
- **CPTAC (Clinical Proteomic Tumor Analysis Consortium)**: Multi-omic cancer data

The data is formatted as dictionaries where:
- **Keys**: Gene identifiers (e.g., Ensembl gene IDs)
- **Values**: Gene expression levels (typically log-transformed counts or TPM values)

## Prerequisites

### Required Libraries
```python
from clinical_transformer import vnBertTokenizer
import pickle
import torch
```

### Data Requirements
- Gene list: A collection of gene identifiers used as vocabulary
- RNA-seq data: Formatted as list of dictionaries with gene:expression pairs

## Step-by-Step Implementation

### 1. Load Gene Vocabulary
```python
# Load the gene list that will serve as the model's vocabulary
# This list defines which genes the model will be trained on
genes = pickle.load(open('/path/to/your/data/gene_list.pk', 'rb'))
```

**What this does:**
- Loads a curated list of genes that will form the model's vocabulary
- These genes are typically selected based on criteria such as:
  - Expression variability across samples
  - Biological relevance
  - Data quality metrics

### 2. Initialize and Configure Tokenizer
```python
# Initialize the vnBERT tokenizer
# This tokenizer is specifically designed to handle continuous gene expression values
tokenizer = vnBertTokenizer()

# Fit the tokenizer on the gene features
# This step creates the vocabulary mapping and sets up value encoding parameters
gene_list = list(genes)
tokenizer.fit(gene_list)
```

**What this does:**
- Creates a vnBERT tokenizer instance optimized for genomic data
- Fits the tokenizer to learn the gene vocabulary and expression value distributions
- Establishes mappings between gene names and token IDs

### 3. Save Pretrained Tokenizer
```python
# Save the fitted tokenizer for future use
# This allows consistent tokenization across training and inference
tokenizer.save_pretrained('/path/to/your/models/tokenizers/ensembl_vnBERT/')
```

**What this does:**
- Persists the tokenizer configuration and vocabulary
- Enables reproducible tokenization for model deployment
- Stores normalization parameters for value encoding

### 4. Load and Filter Training Data
```python
# Load the RNA-seq training dataset
# Data format: List of dictionaries, each representing one sample
X_train = pickle.load(open('/path/to/your/data/RNASeq+tcga+cptac.pk', 'rb'))

# Filter samples to only include genes present in our vocabulary
# This ensures consistency between tokenizer vocabulary and training data
X_train = [{k:v for k,v in i.items() if genes.get(k, False)} for i in X_train]
```

**What this does:**
- Loads the combined TCGA + CPTAC RNA-seq dataset
- Filters each sample to only include genes that are in the model vocabulary
- Ensures data consistency and removes genes not relevant for training

### 5. Tokenize Training Data
```python
# Tokenize the filtered training data
# Configure tokenization parameters for vnBERT training
encoded = tokenizer(
    X_train, 
    return_tensors=None,           # Return as lists rather than tensors
    return_attention_mask=False,   # Don't generate attention masks
    return_quantile_values=False,  # Don't return quantile-normalized values
    return_robust_zscore_values=True,  # Use robust z-score normalization
)
```

**Tokenization Parameters:**
- `return_tensors=None`: Returns Python lists for flexibility in downstream processing
- `return_attention_mask=False`: Attention masks not needed for this training setup
- `return_quantile_values=False`: Skips quantile normalization
- `return_robust_zscore_values=True`: Applies robust z-score normalization to handle outliers

**What robust z-score normalization does:**
- Normalizes expression values using median and median absolute deviation (MAD)
- More robust to outliers compared to standard z-score normalization
- Formula: (x - median) / (MAD)

### 6. Save Processed Dataset
```python
# Save the tokenized dataset for training
# This preprocessed data can be directly used for model training
pickle.dump(encoded, open('/path/to/your/data/GeneTokenizer+vnBERT+RNAseq+tcga+cptac.pk', 'wb'))
```

**What this does:**
- Saves the tokenized and normalized dataset
- Creates a training-ready dataset that can be loaded efficiently during model training
- Preserves the preprocessing pipeline for reproducibility

## Complete Training Pipeline
After data preprocessing, the vnBERT model training involves several key components that work together to create a robust foundation model for genomic data.

### Model Architecture Components

#### 1. Custom Embeddings (`CTEmbeddings`)
vnBERT uses a specialized embedding layer that combines:
- **Token Embeddings**: Standard learnable embeddings for gene tokens
- **Value Embeddings**: Linear projection of continuous expression values
- **Layer Normalization**: Separate normalization for token and value embeddings
- **Scaling**: Square root scaling based on hidden dimension size

#### 2. Transformer Encoder
- Based on BERT architecture with multi-head attention
- Supports FlashAttention for memory efficiency
- Configurable number of layers and attention heads
- Custom attention masking for value-based tokens

#### 3. Value Prediction Head
- Linear layer that predicts masked expression values
- Trained using Mean Squared Error (MSE) loss
- Only computes loss on masked positions

## Complete Training Example

Here's a comprehensive example that demonstrates the entire vnBERT training pipeline from data preparation to model training:

### 1. Configuration Setup

```yaml
# config.yaml - Training configuration file
experiment:
  name: "vnBERT_96M_pancancer"
  version: 1
  seed: 42
  save_dir: "/path/to/experiments"
  
model:
  vocab_size: 20000  # Set based on your gene vocabulary
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 4096
  enable_flash_attention: true
  
dataset:
  input_file: "/path/to/tokenized_data.pk"
  batch_size: 16
  context_window: 2048
  masking_fraction: 0.15
  shuffle: true
  num_workers: 4
  
trainer:
  epochs: 10
  learning_rate: 1e-4
  optimizer: "torch.optim.AdamW"
  strategy: "deepspeed"
  precision: "16-mixed"
  devices: 8
  accumulate_grad_batches: 4
```

### 2. Training Script

```bash
torchrun --nproc_per_node=8 --no-python train_vnBERT config.yaml
```

## Troubleshooting Common Issues

### Memory Issues
- **Problem**: CUDA out of memory errors
- **Solutions**:
  - Reduce `batch_size` in configuration
  - Decrease `context_window` size
  - Enable gradient accumulation
  - Use DeepSpeed with model sharding


## Model Outputs and Applications

The trained vnBERT model produces rich representations that can be used for various downstream tasks:

### Extract Gene Embeddings
```python
# Get contextualized gene representations
with torch.no_grad():
    outputs = model(tokens=sample_tokens, values=sample_values)
    gene_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
```

### Fine-tune for Classification
```python
# Add classification head for cancer type prediction
class vnBERTClassifier(torch.nn.Module):
    def __init__(self, vnbert_model, num_classes):
        super().__init__()
        self.vnbert = vnbert_model
        self.classifier = torch.nn.Linear(768, num_classes)
        
    def forward(self, tokens, values):
        outputs = self.vnbert(tokens=tokens, values=values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(cls_embedding)
```


## Model Release and Production Deployment

After training your vnBERT model, you'll want to prepare it for production use and potentially release it to Hugging Face Hub for broader accessibility. This section covers the complete workflow for converting trained models to production-ready formats.

### Overview of Production Deployment

Production deployment involves several key steps:
1. **Model Conversion**: Transform distributed training checkpoints to single-file format
2. **Weight Extraction**: Convert from Lightning/DeepSpeed format to standard PyTorch
3. **Configuration Packaging**: Prepare model config and tokenizer files
4. **Hugging Face Integration**: Upload to Hugging Face Hub for API access
5. **Production Setup**: Deploy for inference in production environments

### Step 1: Environment Setup and Dependencies

```python
# Required imports for model conversion and deployment
from transformers.models.bert.modeling_bert import BertLayer, BertConfig
from clinical_transformer.pt.datasets.preprocessor.tabular import Preprocessor as Tokenizer
from clinical_transformer.pt.datasets.dataloader.tabular import TabularDataset
from clinical_transformer.pt.training.BERT.nBERT import CTBERT as clinical_transformer

from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.utils.tensor_fragment import fragment_address

import torch
import os
from transformers import BertConfig
```

### Step 2: Configure Deployment Paths

```python
# Configuration for model conversion and deployment
device = 'cuda'

# Source paths - where your trained model is stored
path = '/path/to/your/experiments/'
fm_name = 'ssl/vnBERT'  # Update to match your vnBERT model name
epoch = 47999  # Specify the epoch checkpoint to use
version = 1

# Production destination - where the production-ready model will be stored
production_path = '/path/to/your/production/vnBERT_pancancer_96M_v1.0/'

# Create production directory if it doesn't exist
os.makedirs(production_path, exist_ok=True)
```

**Path Configuration Details:**
- `path`: Root directory of your training experiment
- `fm_name`: Model name within the experiment (should match your training config)
- `epoch`: Specific checkpoint epoch to convert (use best performing epoch)
- `version`: Experiment version number
- `production_path`: Destination for production-ready model files

### Step 3: Convert DeepSpeed Checkpoint to Standard Format

```python
# Register safe globals for checkpoint loading
# This is required for loading DeepSpeed checkpoints with custom classes
torch.serialization.add_safe_globals([
    ZeroStageEnum, 
    LossScaler, 
    BertConfig, 
    fragment_address
])

# Convert distributed DeepSpeed checkpoint to single-file FP32 format
print("Converting DeepSpeed checkpoint to FP32 state dict...")
convert_zero_checkpoint_to_fp32_state_dict(
    checkpoint_dir=f"{path}/models/{fm_name}/version_{version}/models/epoch={epoch}.ckpt",
    output_file=f"{path}/models/{fm_name}/version_{version}/models/epoch={epoch}.ckpt/lightning_model.pt"
)
print("Checkpoint conversion completed successfully!")
```

**What this does:**
- Converts distributed training checkpoints to a single file
- Handles DeepSpeed Zero optimization state reconstruction
- Converts from mixed precision to FP32 for broader compatibility
- Creates a standard PyTorch state dict that can be loaded without DeepSpeed

### Step 4: Package Production Model Files

```python
# Copy converted model weights to production directory
print("Copying model files to production directory...")

# Copy model weights
os.system(f'cp {path}/models/{fm_name}/version_{version}/models/epoch={epoch}.ckpt/lightning_model.pt {production_path}/weights.pt')

# Copy model configuration
os.system(f'cp {path}/models/{fm_name}/version_{version}/model_config.json {production_path}/')

# Copy tokenizer configuration
os.system(f'cp {path}/models/{fm_name}/preprocessor.yaml {production_path}/tokenizer.yaml')

print("Production model packaging completed!")
```

**Production Package Contents:**
- `weights.pt`: Model weights in standard PyTorch format
- `model_config.json`: BERT configuration (vocab size, hidden dims, etc.)
- `tokenizer.yaml`: Tokenizer configuration and vocabulary mapping
