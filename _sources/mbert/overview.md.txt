# mBERT Overview

mBERT (Masked BERT) is the next-generation transformer architecture in Clinical Transformer. It replaces HuggingFace's `BertLayer` with custom SDPA-based layers that call `torch.nn.functional.scaled_dot_product_attention` directly, enabling automatic kernel selection (flash, memory-efficient, or math) without wrapper overhead. It optionally supports Flash Attention 2 (FA2) varlen for zero-padding-waste training on variable-length sequences.

mBERT is designed for:

- Tabular clinical data (mixed categorical and numerical columns)
- Gene expression data (RNA-seq, proteomics)
- Any dataset with continuous and/or categorical features
- Variable-length sequences with dynamic padding

## How It Works

### Pretraining Objective

mBERT is pretrained with **masked value prediction (MVP)**. During each training step:

1. A subset of features is randomly sampled (the **context window**)
2. The last fraction of those features is masked (typically 30%)
3. The model predicts the **value** of each masked feature using the unmasked context
4. The loss is **MSE** computed only over masked positions

### Architecture

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
              │ N × SDPTransformer  │
              │   Layer (SDPA/FA2   │
              │   + feed-forward)   │
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

- A **token embedding** (`nn.Embedding`) maps the feature ID to a vector of size `hidden_size`
- A **value embedding** (`nn.Linear(1, hidden_size)`) projects the scalar value into the same space
- Both are independently layer-normalised, added together, scaled by `√hidden_size`, and passed through a final layer norm

**Encoder** -- a stack of custom `SDPTransformerLayer` modules. Each layer contains:

- `SDPAttention`: multi-head attention calling `F.scaled_dot_product_attention` directly
- Post-LN residual connection + dropout
- Feed-forward network (Linear → GELU → Linear) with post-LN residual

The layers bypass HuggingFace's attention wrapper, which allows arbitrary attention masks (e.g. scGPT 4D mask) to work correctly with SDPA.

**Value prediction head** -- a single `nn.Linear(hidden_size, 1)` that maps each position's hidden state to a predicted scalar value. During training, the prediction head runs only on masked positions for efficiency.

### Attention Backends

mBERT supports two attention backends, controlled by the `attention_backend` config field:

| Backend | Config | When to use | Kernel |
|---------|--------|-------------|--------|
| **SDPA** | `attention_backend: 'sdpa'` | Default. Works on all GPUs. | Flash (no mask) or memory-efficient (with mask) |
| **FA2** | `attention_backend: 'fa2'` | Ampere/Hopper GPUs with variable-length sequences. Requires `pip install flash-attn`. | FA2 varlen (zero padding waste) |

The FA2 backend has a triple fallback guard:
- If `flash-attn` is not installed → falls back to SDPA
- If `use_scgpt_mask: true` → falls back to SDPA (FA2 doesn't support arbitrary masks)
- Otherwise → uses `flash_attn_varlen_func` on packed sequences

### Attention Masking Strategy

mBERT supports two masking modes via the `use_scgpt_mask` config field:

**Full bidirectional** (`use_scgpt_mask: false`) -- every token attends to every other non-padding token. When no padding exists, the attention mask is `None` and SDPA uses the flash kernel. This is the recommended mode for maximum speed.

**scGPT masking** (`use_scgpt_mask: true`) -- inspired by [scGPT](https://www.nature.com/articles/s41592-024-02201-0), this prevents information leakage between masked positions:

- **Unmasked → Unmasked**: ALLOWED (full bidirectional)
- **Unmasked → Masked**: BLOCKED (no leakage from unknowns)
- **Masked → Unmasked**: ALLOWED (gather context for prediction)
- **Masked → Masked**: BLOCKED (no cross-leakage)
- **Masked → Self**: ALLOWED (attend to own token embedding)

Since masking is always a suffix (last N positions), the mask pattern is identical across all samples in a batch. mBERT builds the mask from a single sample `(1, 1, S, S)` and broadcasts, avoiding O(B×S²) memory allocation.

:::{note}
The scGPT masking strategy has not been empirically validated against full bidirectional in ablation studies. Competing single-cell models (scBERT, Geneformer, scFoundation) use full bidirectional attention with competitive results. We provide both options so you can experiment.
:::

### Variable-Length Sequence Support

mBERT handles variable-length sequences through a custom collate function:

```python
from clinical_transformer.mbert.dataset import collate_variable_length

dataloader = DataLoader(
    dataset,
    batch_size=64,
    collate_fn=collate_variable_length,  # pads to max length in batch
)
```

The collate function pads shorter sequences with `token=0`, `value=0.0`, `label=0.0`. The model's padding mask (`tokens != 0`) handles these automatically. When all sequences are the same length (fixed `context_window`), the collate is a no-op.

With the FA2 backend, padded sequences are unpacked before attention (`unpad_input`) and repacked after (`pad_input`), so no compute is wasted on padding tokens.

### Special Tokens

| Token | ID | Value | Purpose |
|-------|-----|-------|---------|
| `<pad>` | 0 | -- | Padding (ignored via attention mask) |
| `<mask>` | 1 | -- | Reserved mask token |
| `<cls>` | 2 | 1.0 | Prepended to every sequence; its hidden state serves as a global representation |

Masked positions are **not** replaced with the `<mask>` token ID. Instead, the original token ID is kept and only the value is set to `-10.0`.

### Dataset Classes

Four dataset implementations handle different data sources:

| Class | Source | Use Case |
|-------|--------|----------|
| `MaskedTokenDataset` | In-memory lists | Small datasets that fit in RAM |
| `MaskedTokenDatasetFromPytorchObject` | PyTorch tensor | Sparse/dense tensor input |
| `MaskedTokenDatasetFromAnnData` | `.h5ad` file (disk-backed) | Large datasets via memory-mapped access |
| `MaskedPriorTokenDataset` | `.h5ad` + biological priors | Gene selection guided by pathway annotations |

All four share the same processing pipeline:
1. Extract features for the sample
2. Sample up to `context_window` features (randomly or via biological priors)
3. Mask the last `mask_prob` fraction of the sequence (values set to `-10.0`)
4. Prepend the CLS token (ID=2, value=1.0)
5. Return `{tokens, values, labels}` tensors

### Training Optimizations

mBERT includes several performance optimizations:

- **Prediction head on masked positions only**: `training_step` skips the `value_predictor` linear layer on non-masked positions (~70-85% of tokens)
- **Broadcast attention mask**: scGPT mask is built from one sample and broadcast over the batch
- **No-mask SDPA**: full bidirectional with no padding passes `attn_mask=None`, enabling the flash kernel
- **Pre-converted dataset**: token/value arrays are converted to tensors once at `__init__`, not per-sample
- **`torch.compile` on inner model**: compiles the model (not the Lightning wrapper) to avoid graph breaks

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
AnnData sparse matrix (.h5ad) or pickle
    |
    v
MaskedTokenDataset ──> samples context window, applies masking
    |
    v
collate_variable_length ──> pads batch to max length
    |
    v
LightningTrainerModel ──> SDPTransformerLayer encoder (SDPA or FA2)
    |                       + value prediction head (MSE loss, masked only)
    v
Trained checkpoint ──> convert with release_model.py
    |
    v
HuggingFace-compatible model ──> inference / embeddings
```

## End-to-End Pipeline

| Step | Page | What It Does |
|------|------|--------------|
| 1 | {doc}`build-dataset` | Tokenize raw data into an AnnData `.h5ad` file |
| 2 | {doc}`configuration` | Write the `config.yaml` that controls training |
| 3 | {doc}`training` | Launch distributed training |
| 4 | {doc}`release-model` | Convert checkpoints to HuggingFace format |
| 5 | {doc}`inference` | Load the model and extract embeddings |
