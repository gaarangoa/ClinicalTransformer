# Step 5: Inference

Once you have a released model from {doc}`release-model`, you can load it and extract embeddings from new data.

## Load the Model

```python
from clinical_transformer import vnBertPretrainedModelForMVP, vnBertTokenizerTabular
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "/path/to/model_hub/"
tokenizer = vnBertTokenizerTabular.from_pretrained(model_path)
model = vnBertPretrainedModelForMVP.from_pretrained(
    model_path, attn_implementation="sdpa"
).to(device)
model.eval()
```

## Prepare Input Data

Each sample is a dictionary mapping feature names to values. The feature names **must match** those used during tokenizer fitting in {doc}`build-dataset`:

```python
sample_data = [
    {
        "Cancer_type": "NSCLC",
        "Drug_class": "PD1",
        "Age": 65.0,
        "BMI": 24.5,
        "NLR": 3.2,
        "TMB": 12.0,
        "Sex": 1,
    },
    # Add more samples...
]
```

## Single-Batch Inference

```python
# Tokenize
processed = tokenizer(
    sample_data,
    return_attention_mask=False,
    return_minmax_values=True,
    return_tensors="pt",
)

# Prepend CLS token (token_id=2, value=1.0)
batch_size = processed["input_ids"].shape[0]
cls_tokens = torch.full((batch_size, 1), 2, dtype=torch.long)
cls_values = torch.full((batch_size, 1), 1.0)

tokens = torch.cat([cls_tokens, processed["input_ids"]], dim=1).to(device)
values = torch.cat([cls_values, processed["minmax_values"]], dim=1).to(device)

# Forward pass
with torch.no_grad():
    output = model(
        tokens=tokens,
        values=values,
        output_last_states=True,
        output_predictions=True,
        return_dict=True,
    )
```

## Extracting Embeddings

There are two common ways to obtain a sample-level embedding:

**CLS embedding** &mdash; the representation from the special classification token:

```python
cls_embeddings = output.last_hidden_state[:, 0, :].detach().cpu()
# Shape: (batch_size, hidden_size)
```

**Mean-pooled embedding** &mdash; the average of all feature embeddings (excludes CLS):

```python
sample_embeddings = (
    output.last_hidden_state[:, 1:, :].mean(dim=1).detach().cpu()
)
# Shape: (batch_size, hidden_size)
```

## Batch Inference on a Full Dataset

For larger datasets, wrap the encoding in a `Dataset` and process in batches:

```python
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.encoded = tokenizer(
            samples,
            return_attention_mask=False,
            return_minmax_values=True,
            return_tensors=None,
        )

    def __len__(self):
        return len(self.encoded["input_ids"])

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded["input_ids"][idx]),
            torch.tensor(self.encoded["minmax_values"][idx]),
        )


dataset = SimpleDataset(sample_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

all_embeddings = []
model.eval()
with torch.no_grad():
    for input_ids, minmax_vals in dataloader:
        bs = input_ids.shape[0]
        cls_t = torch.full((bs, 1), 2, dtype=torch.long)
        cls_v = torch.full((bs, 1), 1.0)
        tokens = torch.cat([cls_t, input_ids], dim=1).to(device)
        values = torch.cat([cls_v, minmax_vals], dim=1).to(device)

        output = model(
            tokens=tokens,
            values=values,
            output_last_states=True,
            return_dict=True,
        )
        emb = output.last_hidden_state[:, 0, :].detach().cpu()
        all_embeddings.append(emb)

import numpy as np

all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
print(f"Final embeddings shape: {all_embeddings.shape}")
```

## What to Do with Embeddings

The extracted embeddings can be used for any downstream task:

- **Classification**: train a linear classifier or MLP on top of the embeddings
- **Clustering**: apply UMAP, t-SNE, or k-means to discover patient subgroups
- **Survival analysis**: use embeddings as features in Cox regression or other survival models
- **Similarity search**: compute cosine similarity between samples
