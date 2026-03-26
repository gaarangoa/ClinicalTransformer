# dBERT Overview

dBERT (Disentangled BERT) is a transformer architecture for continuous-valued data that improves gene embedding quality through **disentangled masked prediction**. It builds on the same base as vnBERT (token + value embeddings, MSE loss on masked positions) but introduces a novel attention masking strategy that creates two simultaneous learning pathways during pretraining.

## Key Difference from vnBERT

vnBERT uses a single masking strategy (scGPT-style) where all masked tokens see known context. dBERT splits masked positions into two groups:

- **Isolated positions**: can only attend to themselves, forcing the embedding table to encode gene behaviour from identity alone
- **Contextual positions**: attend to all known tokens (same as scGPT), training the attention mechanism to learn gene-gene relationships

This disentanglement acts as an implicit regulariser on the embedding space, producing tighter and better-separated gene clusters without any architectural changes or additional parameters.

## How It Works

### Pretraining Objective

dBERT is pretrained with **masked value prediction (MVP)**, identical to vnBERT:

1. A subset of features is randomly sampled (the **context window**)
2. A fraction of those features is masked (typically 15%)
3. The model predicts the **value** of each masked feature
4. The loss is **MSE** computed only over masked positions

The difference is in the attention mask applied during step 3.

### Disentangled Attention Mask

The `mask1_ratio` parameter (default 0.25) controls what fraction of masked positions are isolated:

|              | -> Known | -> Masked1 | -> Masked2 | -> Self |
|--------------|----------|------------|------------|---------|
| **Known**    | yes      | no         | no         | yes     |
| **Masked1**  | no       | no         | no         | yes     |
| **Masked2**  | yes      | no         | no         | yes     |

- `mask1_ratio = 0.0`: recovers the scGPT masking strategy (all masked tokens see context)
- `mask1_ratio = 0.25`: recommended default (25% isolated, 75% contextual)
- `mask1_ratio = -1.0`: standard BERT all-to-all attention (no masking of masked positions)

### Architecture

dBERT uses custom SDPA-based transformer layers (not HuggingFace BertLayer):

```
                  +-------------+
                  |  Input      |
                  |  (token_id, |
                  |   value)    |
                  +------+------+
                         |
              +----------+----------+
              |                     |
     +--------v--------+   +-------v--------+
     |  nn.Embedding    |   |  nn.Linear(1,H)|
     |  (vocab -> H)    |   |  (value -> H)  |
     +--------+---------+   +-------+--------+
              |                     |
       +------v------+      +------v------+
       |  LayerNorm   |      |  LayerNorm   |
       +------+------+      +------+------+
              |                     |
              +----------+----------+
                         |  add
                         v
                    x sqrt(hidden_size)
                         |
                  +------v------+
                  |  LayerNorm   |
                  +------+------+
                         |
              +----------v----------+
              |  N x SDPTransformer  |
              |  Layer (SDPA attn +  |
              |  FFN, post-LN)       |
              +----------+----------+
                         |
                  +------v------+
                  |  LayerNorm   |
                  +------+------+
                         |
                  +------v------+
                  |  Linear(H,1) |  <- value prediction head
                  +------+------+
                         |
                    predicted value
```

### Special Tokens

| Token | ID | Value | Purpose |
|-------|-----|-------|---------|
| `<pad>` | 0 | -- | Padding (ignored via attention mask) |
| `<mask>` | 1 | -- | Reserved mask token |
| `<cls>` | 2 | 1.0 | Prepended to every sequence; global representation |

Masked positions keep their original token ID; only the value is set to `-10.0`.

### Tokenizers

dBERT ships with the same two tokenizers as vnBERT:

**`dBertTokenizer`** -- for gene expression data. Each gene name becomes a vocabulary entry.

**`dBertTokenizerTabular`** -- for mixed tabular data with categorical and numerical columns.

Both support `save_pretrained()` / `from_pretrained()`.

### Dataset Classes

| Class | Source | Use Case |
|-------|--------|----------|
| `MaskedTokenDataset` | In-memory lists | Small datasets that fit in RAM |
| `MaskedTokenDatasetFromPytorchObject` | PyTorch tensor | Sparse/dense tensor input |
| `MaskedTokenDatasetFromAnnData` | `.h5ad` file (disk-backed) | Large datasets via memory-mapped access |
| `MaskedPriorTokenDataset` | `.h5ad` + biological priors | Gene selection guided by pathway annotations |

### Config Parameters Specific to dBERT

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mask1_ratio` | `0.25` | Fraction of masked tokens that are isolated. 0.0 = scGPT, -1.0 = BERT |
| `mask1_warmup_epochs` | `0` | Epochs to linearly ramp mask1_ratio from 0 to target. 0 = no annealing |
| `gated_attention` | `false` | Enable per-head sigmoid gate after SDPA (experimental) |

### Data Flow

```
Raw data (CSV / Excel / DataFrame)
    |
    v
Tokenizer.fit() --> learns vocabulary + per-feature statistics
    |
    v
Tokenizer() --> encodes samples to (token_ids, normalised_values)
    |
    v
AnnData sparse matrix (.h5ad)
    |
    v
MaskedTokenDataset --> samples context window, applies masking
    |
    v
LightningTrainerModel --> dBERT encoder + disentangled attention + value prediction head
    |
    v
Trained checkpoint --> convert with release_model.py
    |
    v
HuggingFace-compatible model --> inference / embeddings
```

## End-to-End Pipeline

| Step | Page | What It Does |
|------|------|--------------|
| 1 | {doc}`build-dataset` | Tokenize raw data into an AnnData `.h5ad` file |
| 2 | {doc}`configuration` | Write the `config.yaml` that controls training |
| 3 | {doc}`training` | Launch distributed training with DeepSpeed |
| 4 | {doc}`release-model` | Convert checkpoints to HuggingFace format |
| 5 | {doc}`inference` | Load the model and extract embeddings |
