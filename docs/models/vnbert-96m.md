# vnBERT 96M v1.0

**Architecture:** vnBERT (Value-based BERT) | **Parameters:** 96M | **Application:** Bulk RNA-seq

## Description

This model implements a variation of the scGPT pretraining strategy where a random set of 1,000 genes is selected and 30% of the expression values are predicted by looking at the remaining genes and their values. Masked genes do not attend to other masked genes during training.

Values are MAD (Median Absolute Deviation) normalised expression scores.

## Quick Start

```python
from clinical_transformer import vnBertPretrainedModel, vnBertTokenizer
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "/path/to/models/bulkRNA_pancancer_vnBERT_96M_v1.0/"
tokenizer = vnBertTokenizer.from_pretrained(model_name)
model = vnBertPretrainedModel.from_pretrained(
    model_name, attn_implementation="sdpa"
).to(device)

sample_data = {
    "ENSG00000001167": 5.6,
    "ENSG00000001460": 0.5,
    "ENSG00000001561": 5.9,
    "ENSG00000001617": 9.4,
}

processed_data = tokenizer(
    [sample_data],
    return_attention_mask=False,
    return_robust_zscore_values=True,
    return_tensors="pt",
)

tokens = torch.cat(
    [torch.tensor([[2]]), processed_data["input_ids"]], dim=1
).to(device)
values = torch.cat(
    [torch.tensor([[1.0]]), processed_data["robust_zscore_values"]], dim=1
).to(device)

model.eval()
with torch.no_grad():
    output = model(
        tokens=tokens,
        values=values,
        output_last_states=True,
        output_predictions=True,
        return_dict=True,
    )
    sample_embeddings = output.last_hidden_state[:, 1:, :].mean(dim=1).detach().cpu()
    cls_embeddings = output.last_hidden_state[:, 0, :].detach().cpu()
```

## Specifications

| Property | Value |
|----------|-------|
| Released checkpoint | 6,399 iterations |
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
