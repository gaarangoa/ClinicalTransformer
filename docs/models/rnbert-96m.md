# rnBERT 96M v1.0

**Architecture:** rnBERT (Ranked Normalised BERT) | **Parameters:** 96M | **Application:** Bulk RNA-seq

## Description

Ranked nBERT uses sequential input where values are ranked positions normalised from 0&ndash;1. It selects a context window of consecutive genes and predicts masked genes by looking at neighbouring genes and their ranked values. Implements the HuggingFace API.

## Quick Start

```python
from clinical_transformer import rnBertPretrainedModel, rnBertTokenizer
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "/path/to/models/bulkRNA_pancancer_rnBERT_96M_v1.0/"
tokenizer = rnBertTokenizer.from_pretrained(model_name)
model = rnBertPretrainedModel.from_pretrained(
    model_name, attn_implementation="eager"
).to(device)

sample_data = {
    "ENSG00000001167": 5.6,
    "ENSG00000001460": 0.5,
    "ENSG00000001561": 5.9,
    "ENSG00000001617": 9.4,
}

processed_data = tokenizer(
    [sample_data], return_attention_mask=False, return_tensors="pt"
)

tokens = torch.cat(
    [torch.tensor([[2]]), processed_data["input_ids"]], dim=1
).to(device)
values = torch.cat(
    [torch.tensor([[0.0]]), processed_data["gene_values"]], dim=1
).to(device)

model.eval()
with torch.no_grad():
    output = model(
        tokens=tokens,
        values=values,
        output_last_states=True,
        output_predictions=False,
        return_dict=True,
    )
    sample_embeddings = output.last_hidden_state[:, 1:, :].mean(dim=1).detach()
    cls_embeddings = output.last_hidden_state[:, 0, :].detach()
```

## Specifications

| Property | Value |
|----------|-------|
| Released checkpoint | 10,000 iterations |
| Vocabulary | 20,000 genes |
| Embedding size | 1,024 |
| Transformer layers | 12 |
| Attention heads | 16 |
| Feed-forward size | 1,024 |
| Context window | 1,000 genes |
| Training framework | DeepSpeed |

## Training Data

- **TCGA** (The Cancer Genome Atlas)
- **CPTAC** (Clinical Proteomic Tumor Analysis Consortium)
