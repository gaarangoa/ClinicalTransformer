# BulkRNA pancancer vnBERT_96M_v1.0

This guide provides comprehensive instructions for using the value prediction model vnBERT v1.0 foundation model for bulk RNA-seq data inference and analysis.

This model implements a variation of the scGPT pretraining strategy where a random set of 1,000 genes is selected and 30% the expression value of those are genes predicted by looking at other genes and values. Note that in this model the attention is build so masked genes do not attend to other masked genes. 

The model uses as values the MAD normalized expression scores. 

## Prerequisites
```bash
# Install required dependencies
pip install git+https://github.com/<your-org>/clinical_transformer.git@master  # Main package
```

## Usage
```python
from clinical_transformer import vnBertPretrainedModel, vnBertTokenizer
import torch 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_name = "/path/to/your/models/bulkRNA_pancancer_vnBERT_96M_v1.0/"
tokenizer = vnBertTokenizer.from_pretrained(model_name)
model = vnBertPretrainedModel.from_pretrained(model_name, attn_implementation="sdpa").to(device)

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
processed_data = tokenizer([sample_data], return_attention_mask=False, return_robust_zscore_values=True, return_tensors='pt')

tokens = torch.cat([torch.tensor([[2]]), processed_data['input_ids']], dim=1).to(device)
values = torch.cat([torch.tensor([[1.0]]), processed_data['robust_zscore_values']], dim=1).to(device)

model.eval()
with torch.no_grad():
    output = model(tokens=tokens, values=values, output_last_states=True, output_predictions=True, return_dict=True)
    sample_embeddings = output.last_hidden_state[:, 1:, :].mean(dim=1).detach().cpu()
    cls_embeddings = output.last_hidden_state[:, 0, :].detach().cpu()
```

## Model Overview
### Released Model Specifications
- **Model Checkpoint**: `6399` iterations
- **Architecture**: nvBERT (normalized value predictor BERT Clinical Transformer)
- **Model Size**: 96M parameters
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
