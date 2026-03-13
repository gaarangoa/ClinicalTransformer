# nBERT 80M v1.0

**Architecture:** nBERT (Normalised BERT) | **Parameters:** 80M | **Application:** Bulk RNA-seq

## Description

nBERT is trained to predict masked **gene names** by looking at its value and randomly selected genes and values.

## Quick Start

```python
import torch
from clinical_transformer import nBertPretrainedModel, nBertTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "/path/to/models/bulkRNA_pancancer_nBERT_80M_v1.0/"
tokenizer = nBertTokenizer.from_pretrained(model_name)
model = nBertPretrainedModel.from_pretrained(model_name).to(device)

sample_data = {
    "ENSG00000001167": 5.6,
    "ENSG00000001460": 0.5,
    "ENSG00000001561": 5.9,
    "ENSG00000001617": 9.4,
}

processed_data = tokenizer.transform([sample_data])
token_ids = processed_data[0][0]
expression_values = processed_data[0][1]

tokens = torch.tensor([[2] + token_ids]).to(device)
values = torch.tensor([[1.0] + expression_values]).to(device)

model.eval()
with torch.no_grad():
    outputs = model(
        tokens=tokens,
        values=values,
        output_last_states=True,
        output_attentions=False,
        output_hidden_states=False,
    )

cls_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu()
```

## Specifications

| Property | Value |
|----------|-------|
| Released checkpoint | 10,000 iterations |
| Vocabulary | 20,000 genes |
| Embedding size | 1,024 |
| Transformer layers | 10 |
| Attention heads | 16 |
| Feed-forward size | 1,024 |
| Context window | 1,000 genes |
| Training framework | DeepSpeed |

## Training Data

- **TCGA** (The Cancer Genome Atlas)
- **CPTAC** (Clinical Proteomic Tumor Analysis Consortium)
