# Clinical Transformer

A framework for training and deploying transformer-based foundation models on clinical, genomic, and tabular data. Built on HuggingFace Transformers with distributed training via DeepSpeed and PyTorch Lightning.

## Key Features

- **Value-based BERT (vnBERT)** -- transformer architecture that natively handles continuous numerical values alongside categorical tokens
- **Masked Value Prediction** -- self-supervised pretraining objective that learns feature relationships by predicting masked values from context
- **HuggingFace-compatible** -- trained models are saved in standard HuggingFace format for easy loading and sharing
- **Scalable training** -- multi-GPU distributed training with DeepSpeed ZeRO and mixed precision
- **Flexible tokenizers** -- separate tokenizers for gene expression data and mixed tabular data (numerical + categorical)

## Quick Start

### Installation

```bash
pip install https://github.com/gaarangoa/ClinicalTransformer.git
```

### Inference

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

# Each sample is a dict mapping feature names to values
samples = [{"Age": 65.0, "BMI": 24.5, "Cancer_type": "NSCLC", "TMB": 12.0}]

processed = tokenizer(
    samples,
    return_attention_mask=False,
    return_minmax_values=True,
    return_tensors="pt",
)

# Prepend CLS token
batch_size = processed["input_ids"].shape[0]
tokens = torch.cat(
    [torch.full((batch_size, 1), 2, dtype=torch.long), processed["input_ids"]], dim=1
).to(device)
values = torch.cat(
    [torch.full((batch_size, 1), 1.0), processed["minmax_values"]], dim=1
).to(device)

with torch.no_grad():
    output = model(
        tokens=tokens, values=values,
        output_last_states=True, return_dict=True,
    )
    embeddings = output.last_hidden_state[:, 0, :].detach().cpu()  # CLS embeddings
```

## Training Pipeline

The full vnBERT pipeline has five steps:

| Step | Description |
|------|-------------|
| **1. Build Dataset** | Tokenize raw data (CSV/DataFrame) into an AnnData `.h5ad` file |
| **2. Configuration** | Write a `config.yaml` controlling model architecture, training, and data |
| **3. Training** | Launch distributed pretraining with `torchrun` and DeepSpeed |
| **4. Release Model** | Convert DeepSpeed checkpoints to HuggingFace format |
| **5. Inference** | Load the model and extract embeddings for downstream tasks |

```bash
# Training launch example (8 GPUs)
torchrun --nproc_per_node=8 train.py config.yaml
```

See the [full documentation](docs/) for detailed guides on each step.

## Pretrained Models

| Model | Parameters | Data | Application |
|-------|-----------|------|-------------|
| vnBERT RNA | TBD | TBD | TBD |


## Requirements

- Python 3.10+
- PyTorch 2.5+
- Lightning 2.5+
- DeepSpeed 0.16+
- CUDA-capable GPU (for training)

## License

See [LICENSE](LICENSE) for details.
