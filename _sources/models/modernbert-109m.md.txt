# ModernBERT 109M v1.0

**Architecture:** ModernBERT | **Parameters:** 109M | **Application:** Bulk RNA-seq

## Description

Based on the ModernBERT architecture with RoPE positional encoding. This is a mask prediction model with a context window of 1,000 genes. **No expression values are used** for training, making it invariant to absolute expression levels. Uses the HuggingFace API directly.

## Quick Start

```python
import torch
from clinical_transformer import ModernBertRankTokenizer
from transformers import ModernBertModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "/path/to/models/bulkRNA_pancancer_modernBERT_109M_v1.0/"
tokenizer = ModernBertRankTokenizer.from_pretrained(model_name)
model = ModernBertModel.from_pretrained(model_name).to(device)

sample_data = {
    "ENSG00000001167": 5.6,
    "ENSG00000001460": 0.5,
    "ENSG00000001561": 5.9,
    "ENSG00000001617": 9.4,
}

processed_data = tokenizer.transform([sample_data])
token_ids = torch.tensor([processed_data[0][0]]).to(device)

model.eval()
with torch.no_grad():
    out = model(input_ids=token_ids)
    embeddings = out.last_hidden_state
    sample_embeddings = embeddings.mean(dim=1)
```

## Specifications

| Property | Value |
|----------|-------|
| Released checkpoint | 4,599 iterations |
| Vocabulary | 20,000 genes |
| Embedding size | 1,024 |
| Transformer layers | 12 |
| Attention heads | 16 |
| Feed-forward size | 1,024 |
| Context window | 1,000 genes |
| Positional encoding | RoPE |
| Training framework | DeepSpeed |

## Training Data

- **TCGA** (The Cancer Genome Atlas)
- **CPTAC** (Clinical Proteomic Tumor Analysis Consortium)
