# BulkRNA pancancer modernBERT_109M_v1.0

This guide provides comprehensive instructions for using the modern BERT v1.0 foundation model for bulk RNA-seq data inference and analysis.

This model is based on the <a href="">Modern BERT</a> architecture and uses the hugging face API. This is a mask prediction model with a context window of 1,000 genes and uses RoPE for positinal encoding. No expression values are used for training this model. Therefore making it invariant to absolute expression values. 

## Prerequisites
```bash
# Install required dependencies
pip install git+https://github.com/<your-org>/clinical_transformer.git@master  # Main package
```

## Usage
```python
import torch
import numpy as np
from clinical_transformer import ModernBertRankTokenizer
from transformers import ModernBertModel

# Check CUDA availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define model path
model_name = "/path/to/your/models/bulkRNA_pancancer_modernBERT_109M_v1.0/"

# Load the pre-trained tokenizer
tokenizer = ModernBertRankTokenizer.from_pretrained(model_name)

# Load the pre-trained model
model = ModernBertModel.from_pretrained(model_name).to(device)

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

# check the generated list in ranked order
print([tokenizer.feature_decoder[i] for i in processed_data[0][0]])

# The tokenizer returns [token_ids, values] for each sample
token_ids = torch.tensor([processed_data[0][0]]).to(device)  # List of token IDs

# Perform inference
model.eval()
with torch.no_grad():
    out = model(input_ids = token_ids)
    embeddings = out.last_hidden_state
    sample_embeddings = embeddings.mean(dim=1)
```


## Model Overview
### Released Model Specifications
- **Model Checkpoint**: `4599` iterations
- **Architecture**: nBERT (normalized BERT Clinical Transformer)
- **Model Size**: 109M parameters
- **Application**: Bulk RNA sequencing data analysis

### Model Architecture Details
- **Features**: 20,000 genes
- **Embedding Size**: 1,024 dimensions
- **Transformer Layers**: 12 layers
- **Attention Heads**: 16 heads per layer
- **Feed-Forward Network**: 1,024 hidden units
- **Training Iterations**: 10,000 iterations
- **Context Window**: 1,000 features (genes)
- **Training Framework**: DeepSpeed with model sharding

### Training Data
- **Datasets**: TCGA (The Cancer Genome Atlas) + CPTAC (Clinical Proteomic Tumor Analysis Consortium)
