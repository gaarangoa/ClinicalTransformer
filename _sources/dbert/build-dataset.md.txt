# Step 1: Build the Dataset

dBERT trains on an **AnnData** (`.h5ad`) sparse matrix where rows are samples and columns are token IDs. The `dBertTokenizerTabular` tokenizer converts your raw tabular data into this format.

## 1.1 Load Your Data

Start from any tabular file &mdash; CSV, Excel, Parquet, or anything pandas can read:

```python
import pandas as pd
import numpy as np
import anndata
from scipy.sparse import coo_matrix
from clinical_transformer import dBertTokenizerTabular as TokenizerTabular

# ── EDIT THESE for your dataset ──────────────────────────
model_id = 'MyModel'
DATA_PATH = f'data/{model_id}/my_dataset.csv'
TOKENIZER_SAVE_DIR = f'models/ssl/{model_id}/tokenizer/'
OUTPUT_PATH = f'data/{model_id}/training_data.h5ad'
# ─────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)   # or pd.read_excel(), pd.read_parquet()
print(f'Loaded {df.shape[0]} samples x {df.shape[1]} columns')
```

## 1.2 Define Your Features

Split your columns into three groups:

Categorical features
: String or label columns the model should learn (e.g. diagnosis, treatment, sex).

Numerical features
: Continuous or binary-coded numbers (e.g. age, BMI, lab values, gene expression). Binary 0/1 columns go here too.

Labels / metadata
: Columns kept as `.obs` metadata in the AnnData object. These are **not** tokenised and are not seen by the model during training. Use this for sample IDs, outcomes, survival times, etc.

```python
# ── EDIT THESE for your dataset ──────────────────────────
categorical_features = [
    'Cancer_type',
    'Drug_class',
]

numerical_features = [
    'Age',
    'BMI',
    'NLR',
    'TMB',
    'Sex',
]

label_columns = [
    'SAMPLE_ID',
    'OS_Event',
    'OS_Months',
]
# ─────────────────────────────────────────────────────────

print(f'Categorical: {len(categorical_features)}')
print(f'Numerical:   {len(numerical_features)}')
print(f'Labels/meta: {len(label_columns)}')
```

## 1.3 Fit and Save the Tokenizer

The tokenizer learns two things during `.fit()`:

1. A **vocabulary** mapping each feature name to a unique token ID
2. The **min/max range** for each numerical feature (used for normalisation)

```python
feature_cols = categorical_features + numerical_features
samples = df[feature_cols].to_dict(orient='records')

tokenizer = TokenizerTabular()
tokenizer.fit(
    samples,
    categorical_features=categorical_features,
    numerical_features=numerical_features,
)

tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)
```

:::{important}
Save the tokenizer now. You will need it again when releasing the model (Step 4) and during inference (Step 5). The tokenizer directory must live alongside your training outputs.
:::

## 1.4 Encode and Inspect

Encode all samples. The tokenizer returns:

- `input_ids`: the token ID for each feature
- `minmax_values`: the value normalised to [0, 1] using the learned min/max

```python
encoded = tokenizer(
    samples,
    return_tensors=None,
    return_attention_mask=False,
    return_minmax_values=True,
)

# ── Inspect a single sample ─────────────────────────────
sample_idx = 0
sample_raw = samples[sample_idx]
sample_enc = tokenizer.encode_sample(sample_raw, return_minmax_values=True)

rows_example = []
for i, tid in enumerate(sample_enc['input_ids']):
    feat = tokenizer.convert_ids_to_tokens(tid)
    rows_example.append({
        'feature': feat,
        'token_id': tid,
        'raw_value': sample_raw.get(feat, None),
        'minmax_encoded': round(sample_enc['minmax_values'][i], 4),
    })

example_df = pd.DataFrame(rows_example)
print(example_df.to_string())
```

Verify that each feature maps to a unique token ID and that values are normalised between 0 and 1 before moving on.

## 1.5 Build the Sparse AnnData Matrix

Each sample becomes a sparse row where column indices are token IDs and values are the minmax-encoded numbers:

```python
cols = encoded['input_ids']
values = encoded['minmax_values']

rows = []
for ix, v in enumerate(values):
    rows += [ix] * len(v)

cols_flat = np.concatenate(cols)
values_flat = np.concatenate(values)
rows_flat = np.array(rows)

sparse_matrix = coo_matrix(
    (values_flat, (rows_flat, cols_flat)),
    shape=(len(samples), tokenizer.vocab_size),
)

adata = anndata.AnnData(X=sparse_matrix.tocsr())

# Attach metadata
adata.obs.index = df['SAMPLE_ID'].astype(str).values
for col in label_columns:
    adata.obs[col] = df[col].values

print(adata)
```

## 1.6 Save and Verify

```python
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
adata.write(OUTPUT_PATH)
print(f'Saved to {OUTPUT_PATH}  |  Shape: {adata.shape}')

# Sanity check
adata2 = anndata.read_h5ad(OUTPUT_PATH)
print(f'Reloaded shape: {adata2.shape}')
print(f'Non-zero entries (sample 0): {np.sum(adata2[0].X.toarray() != 0)}')
print(f'Obs columns: {list(adata2.obs.columns)}')
```

:::{tip}
For faster data loading during training, copy the `.h5ad` file to shared memory:

```bash
cp data/MyModel/training_data.h5ad /dev/shm/training_data.h5ad
```

Then point `dataset.input_file` in your config to `/dev/shm/training_data.h5ad`.
:::
