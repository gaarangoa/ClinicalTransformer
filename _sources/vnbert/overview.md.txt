# vnBERT Overview

vnBERT (Value-based BERT) is a transformer architecture that directly handles continuous numerical values alongside categorical tokens. Unlike standard BERT which operates on discrete tokens, vnBERT encodes each feature as a **token + value** pair, making it ideal for:

- Tabular clinical data (mixed categorical and numerical columns)
- Gene expression data (RNA-seq, proteomics)
- Any dataset with continuous and/or categorical features

## How It Works

### Pretraining Objective

vnBERT is pretrained with **masked value prediction (MVP)**. During each training step:

1. A subset of features is randomly sampled (the **context window**)
2. A fraction of those features is masked (typically 30%)
3. The model predicts the **value** of each masked feature using the unmasked context
4. The loss is **MSE** computed only over masked positions

This forces the model to learn meaningful relationships between features.

### Architecture

vnBERT extends HuggingFace's `BertPreTrainedModel` with custom embedding and attention logic:

```
                  ┌─────────────┐
                  │  Input      │
                  │  (token_id, │
                  │   value)    │
                  └──────┬──────┘
                         │
              ┌──────────┴──────────┐
              │                     │
     ┌────────▼────────┐   ┌───────▼────────┐
     │  nn.Embedding    │   │  nn.Linear(1,H)│
     │  (vocab → H)     │   │  (value → H)   │
     └────────┬────────┘   └───────┬────────┘
              │                     │
       ┌──────▼──────┐      ┌──────▼──────┐
       │  LayerNorm   │      │  LayerNorm   │
       └──────┬──────┘      └──────┬──────┘
              │                     │
              └──────────┬──────────┘
                         │  add
                         ▼
                    × √hidden_size
                         │
                  ┌──────▼──────┐
                  │  LayerNorm   │
                  └──────┬──────┘
                         │
              ┌──────────▼──────────┐
              │   N × BertLayer     │
              │   (self-attention +  │
              │    feed-forward)     │
              └──────────┬──────────┘
                         │
                  ┌──────▼──────┐
                  │  LayerNorm   │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  Linear(H,1) │  ← value prediction head
                  └──────┬──────┘
                         │
                    predicted value
```

**Embedding layer (`CTEmbeddings`)** -- each input position has two components:

- A **token embedding** (`nn.Embedding`) maps the feature ID (e.g. gene index or column index) to a vector of size `hidden_size`
- A **value embedding** (`nn.Linear(1, hidden_size)`) projects the scalar value into the same space
- Both are independently layer-normalised, added together, scaled by `√hidden_size`, and passed through a final layer norm

**Encoder** -- a stack of standard `BertLayer` modules (multi-head self-attention + feed-forward) from HuggingFace Transformers. The number of layers, heads, and hidden size are all configurable via `BertConfig`.

**Value prediction head** -- a single `nn.Linear(hidden_size, 1)` that maps each position's hidden state to a predicted scalar value. During pretraining, loss is computed only on masked positions.

### Attention Masking Strategy

vnBERT uses an attention masking strategy inspired by [scGPT](https://www.nature.com/articles/s41592-024-02201-0) to prevent information leakage during masked value prediction:

- **Unmasked tokens** can attend to all other unmasked tokens (normal self-attention)
- **Masked tokens** can only attend to themselves (via diagonal self-attention) -- they cannot see other masked tokens
- **No position can attend to a masked token** as a key/value -- masked tokens are excluded from the key/value set

This is implemented by constructing a custom attention mask per layer:

```python
# Positions with value == -10.0 are masked
masked_positions = (values == -10.0)

# Key/value mask: only unmasked, non-padding tokens can be keys
key_value_mask = padding_mask & (~masked_positions)

# Diagonal: masked tokens attend to themselves
diagonal_mask = torch.eye(seq_len)
masked_self_attend = masked_positions & diagonal_mask

# Final mask = normal attention + masked self-attention
attention_mask = (padding_mask & key_value_mask) | masked_self_attend
```

This ensures the model must predict masked values purely from unmasked context, not by copying from neighbouring masked positions.

### Special Tokens

| Token | ID | Value | Purpose |
|-------|-----|-------|---------|
| `<pad>` | 0 | -- | Padding (ignored via attention mask) |
| `<mask>` | 1 | -- | Reserved mask token |
| `<cls>` | 2 | 1.0 | Prepended to every sequence; its hidden state serves as a global representation |

Masked positions are **not** replaced with the `<mask>` token ID. Instead, the original token ID is kept and only the value is set to `-10.0`, which the model recognises as a masked signal.

### Tokenizers

vnBERT ships with two HuggingFace-compatible tokenizers (`PreTrainedTokenizer` subclasses):

**`vnBertTokenizer`** -- for gene expression data. Each gene name (e.g. `ENSG00000001167`) becomes a vocabulary entry. The `fit()` method learns the vocabulary and computes per-gene statistics (min, max, median, MAD) for robust z-score normalisation.

**`vnBertTokenizerTabular`** -- for mixed tabular data. Handles both numerical and categorical columns:
- Categorical features are ordinal-encoded (each level mapped to an integer) and then normalised like numerical features
- Numerical features are normalised per-feature using min-max, z-score, or robust z-score
- Both output `(input_ids, values)` pairs compatible with the model

Both tokenizers support `save_pretrained()` / `from_pretrained()` for portability.

### Dataset Classes

Four dataset implementations handle different data sources and sampling strategies:

| Class | Source | Use Case |
|-------|--------|----------|
| `MaskedTokenDataset` | In-memory lists | Small datasets that fit in RAM |
| `MaskedTokenDatasetFromPytorchObject` | PyTorch tensor | Sparse/dense tensor input |
| `MaskedTokenDatasetFromAnnData` | `.h5ad` file (disk-backed) | Large datasets via memory-mapped access |
| `MaskedPriorTokenDataset` | `.h5ad` + biological priors | Gene selection guided by pathway annotations |

All four share the same processing pipeline:
1. Extract non-zero features for the sample
2. Sample up to `context_window` features (randomly or via biological priors)
3. Mask the last `mask_prob` fraction of the sequence (values set to `-10.0`)
4. Prepend the CLS token (ID=2, value=1.0)
5. Return `{tokens, values, labels}` tensors

### Training

The `LightningTrainerModel` wraps the model as a PyTorch Lightning module:

- **Optimizer**: configurable via config (supports `DeepSpeedCPUAdam` and `FusedAdam`)
- **Loss**: `nn.MSELoss(reduction='none')` computed only on positions where `values == -10.0`
- **Strategy**: DeepSpeed ZeRO for distributed multi-GPU training
- **Checkpointing**: periodic saves via Lightning's `ModelCheckpoint`

### Data Flow

```
Raw data (CSV / Excel / DataFrame)
    |
    v
Tokenizer.fit() ──> learns vocabulary + per-feature statistics
    |
    v
Tokenizer() ──> encodes samples to (token_ids, normalised_values)
    |
    v
AnnData sparse matrix (.h5ad)
    |
    v
MaskedTokenDataset ──> samples context window, applies masking
    |
    v
LightningTrainerModel ──> BERT encoder + value prediction head (MSE loss)
    |
    v
Trained checkpoint ──> convert with release_model.py
    |
    v
HuggingFace-compatible model ──> inference / embeddings
```

## End-to-End Pipeline

The full pipeline consists of five steps, each covered in its own page:

| Step | Page | What It Does |
|------|------|--------------|
| 1 | {doc}`build-dataset` | Tokenize raw data into an AnnData `.h5ad` file |
| 2 | {doc}`configuration` | Write the `config.yaml` that controls training |
| 3 | {doc}`training` | Launch distributed training with DeepSpeed |
| 4 | {doc}`release-model` | Convert checkpoints to HuggingFace format |
| 5 | {doc}`inference` | Load the model and extract embeddings |

## Project Structure

We recommend organising your experiment like this:

```
my_experiment/
├── data/
│   └── MyModel/
│       ├── my_dataset.csv          # raw input
│       └── training_data.h5ad     # output of Step 1
├── models/
│   └── ssl/
│       └── MyModel/
│           ├── tokenizer/          # saved tokenizer (Step 1)
│           └── version_1/          # training outputs (Step 3)
│               ├── models/         # checkpoints
│               └── model_config.json
├── model_hub/                      # released model (Step 4)
├── config.yaml
├── build_dataset.py
├── train.py
├── train.sh
└── release_model.py
```
