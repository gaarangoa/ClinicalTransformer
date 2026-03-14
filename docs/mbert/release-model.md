# Step 4: Release the Model

After training, checkpoints need to be converted to a single-file, HuggingFace-compatible format for inference.

## Conversion Script

```python
from clinical_transformer.mbert import mBertPretrainedModelForMVP
from clinical_transformer.mbert import mBertTokenizerTabular as Tokenizer
import torch

from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.utils.tensor_fragment import fragment_address
from transformers import BertConfig

from clinical_transformer.mbert.modeling import LightningTrainerModel

# ── EDIT THESE ───────────────────────────────────────────
model_path = "path/to/model_hub"       # Where the final model will be saved
path = "path/to/models/ssl"            # save_dir from config.yaml
fm_name = "My_FM_NAME"                 # experiment.name from config.yaml
epoch = 1000                           # Epoch checkpoint to release
version = 1                            # experiment.version from config.yaml
# ─────────────────────────────────────────────────────────
```

### Convert the DeepSpeed Checkpoint

This merges all sharded files into a single FP32 state dict:

```python
torch.serialization.add_safe_globals(
    [ZeroStageEnum, LossScaler, BertConfig, fragment_address]
)

convert_zero_checkpoint_to_fp32_state_dict(
    f"{path}/{fm_name}/version_{version}/models/epoch={epoch}.ckpt",
    f"{path}/{fm_name}/version_{version}/models/epoch={epoch}.ckpt/lightning_model.pt",
)
```

### Load and Save as HuggingFace Model

```python
model_config = BertConfig.from_pretrained(
    f"{path}/{fm_name}/version_{version}/model_config.json"
)

model = LightningTrainerModel.load_from_checkpoint(
    f"{path}/{fm_name}/version_{version}/models/epoch={epoch}.ckpt/lightning_model.pt",
    config=model_config,
)

total_params = sum(p.numel() for p in model.model.parameters())
print(f"Total parameters: {total_params:,}")

# Save model weights in HuggingFace format
model.model.save_pretrained(f"{model_path}/")

# Save tokenizer alongside the model
tokenizer = Tokenizer.from_pretrained(
    f"{path}/{fm_name}/tokenizer"
)
tokenizer.save_pretrained(f"{model_path}/")

print(f"Model released to: {model_path}")
```

:::{note}
If you trained with DDP instead of DeepSpeed, skip the `convert_zero_checkpoint_to_fp32_state_dict` step and load the checkpoint directly.
:::

## Output

After running this script, `model_path/` will contain:

```
model_hub/
├── config.json            # BERT configuration
├── model.safetensors      # Model weights
├── tokenizer_config.json  # Tokenizer configuration
└── ...                    # Additional tokenizer files
```

:::{tip}
Pick the checkpoint epoch that shows the best (lowest) training loss. Inspect the loss CSV from Step 3 to find it.
:::
