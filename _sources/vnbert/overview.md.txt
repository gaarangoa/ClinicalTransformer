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
4. Masked features do not attend to other masked features

This forces the model to learn meaningful relationships between features.

### Architecture

vnBERT builds on the HuggingFace BERT architecture with custom modifications:

- **Token embeddings**: standard learnable embeddings mapping feature IDs to vectors
- **Value embeddings**: a linear projection that encodes the continuous value into the same embedding space
- **Combined representation**: token and value embeddings are added, normalised, and scaled
- **Transformer encoder**: standard multi-head self-attention with configurable depth and width
- **Value prediction head**: a linear layer that outputs predicted values for masked positions, trained with MSE loss

### Data Flow

```
Raw data (CSV / Excel / DataFrame)
    |
    v
Tokenizer.fit() ──> learns vocabulary + min/max ranges
    |
    v
Tokenizer() ──> encodes samples to (token_ids, minmax_values)
    |
    v
AnnData sparse matrix (.h5ad)
    |
    v
MaskedTokenDataset ──> samples context window, applies masking
    |
    v
LightningTrainerModel ──> BERT encoder + value prediction head
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
