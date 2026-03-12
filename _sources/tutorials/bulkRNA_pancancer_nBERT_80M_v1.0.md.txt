# BulkRNA pancancer nBERT_80M_v1.0

This guide provides comprehensive instructions for using the nBERT v1.0 foundation model for bulk RNA-seq data inference and analysis.

This model is trained to predict masked genes names by looking at its value and randomly selected genes and values.  

## Prerequisites
```bash
# Install required dependencies
pip install git+https://github.com/<your-org>/clinical_transformer.git@master  # Main package
```

## Usage
```python
import torch
import numpy as np
from clinical_transformer import nBertPretrainedModel, nBertTokenizer

# Check CUDA availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define model path
model_name = '/path/to/your/models/bulkRNA_pancancer_nBERT_80M_v1.0/'

# Load the pre-trained tokenizer
tokenizer = nBertTokenizer.from_pretrained(model_name)

# Load the pre-trained model
model = nBertPretrainedModel.from_pretrained(model_name).to(device)

# The model expects gene expression data as a dictionary mapping Ensembl gene IDs to expression values:
# Example: Single sample gene expression data
# Gene IDs should be Ensembl IDs (ENSG format)
# Expression values should be Log2(TPM+1)
sample_data = {
    "ENSG00000001167": 5.6,  # NFYA gene - high expression
    "ENSG00000001460": 0.5,  # STPG1 gene - low expression  
    "ENSG00000001561": 5.9,  # ENPP4 gene - high expression
    "ENSG00000001617": 9.4,  # SEMA3F gene - very high expression
    # Add more genes as needed...
}

# Transform the data using the tokenizer
processed_data = tokenizer.transform([sample_data])

# The tokenizer returns [token_ids, values] for each sample
token_ids = processed_data[0][0]  # List of token IDs
expression_values = processed_data[0][1]  # List of corresponding expression values

# Add special tokens
# Token ID 2 corresponds to <CLS> (classification token)
# Value 1.0 is used as a placeholder for the <CLS> token
tokens = torch.tensor([[2] + token_ids]).to(device)
values = torch.tensor([[1.0] + expression_values]).to(device)

# Perform inference
model.eval()  # Set to evaluation mode
with torch.no_grad():
    outputs = model(
        tokens=tokens, 
        values=values,
        output_last_states=True,      # Get final hidden states
        output_attentions=False,      # Skip attention weights for speed
        output_hidden_states=False    # Skip intermediate hidden states
    )

# Extract embeddings from the last hidden layer
embeddings = outputs.last_hidden_state.detach().cpu()

# Get the CLS token embedding (first token)
cls_embedding = embeddings[:, 0, :]  # Shape: [batch_size, embedding_dim]
```


## Model Overview
### Released Model Specifications
- **Model Checkpoint**: `10000` iterations
- **Architecture**: nBERT (normalized BERT Clinical Transformer)
- **Model Size**: 80M parameters
- **Application**: Bulk RNA sequencing data analysis

### Model Architecture Details
- **Features**: 20,000 genes
- **Embedding Size**: 1,024 dimensions
- **Transformer Layers**: 10 layers
- **Attention Heads**: 16 heads per layer
- **Feed-Forward Network**: 1,024 hidden units
- **Training Iterations**: 10,000 iterations
- **Context Window**: 1,000 features (genes)
- **Training Framework**: DeepSpeed with model sharding

### Training Data
- **Datasets**: TCGA (The Cancer Genome Atlas) + CPTAC (Clinical Proteomic Tumor Analysis Consortium)
