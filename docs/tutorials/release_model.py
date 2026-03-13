from clinical_transformer import vnBertPretrainedModelForMVP
from clinical_transformer import vnBertTokenizerTabular as vnBertTokenizer
import pickle
import torch

from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.utils.tensor_fragment import fragment_address
from transformers import BertConfig

from clinical_transformer.vnbert.modeling import LightningTrainerModel


model_path = "path_to_model_hub"
device = 'cuda'
path='path_to_pretrained_model' 
fm_name='MyModel'
epoch=1000
version=1



torch.serialization.add_safe_globals([ZeroStageEnum, LossScaler, BertConfig, fragment_address])
convert_zero_checkpoint_to_fp32_state_dict(
    f"{path}/models/{fm_name}/version_{version}/models/epoch={epoch}.ckpt",
    f"{path}/models/{fm_name}/version_{version}/models/epoch={epoch}.ckpt/lightning_model.pt"
);


# Save HF checkpoint
model_config = BertConfig.from_pretrained(f'{path}/models/{fm_name}/version_{version}/model_config.json')
model = LightningTrainerModel.load_from_checkpoint(
    f'{path}/models/{fm_name}/version_{version}/models/epoch={epoch}.ckpt/lightning_model.pt',
    config=model_config
)

total_params = sum(param.numel() for param in model.model.parameters())
print(f"Total parameters: {total_params}")

model.model.save_pretrained(f'{model_path}/')
tokenizer= vnBertTokenizer.from_pretrained(f"{path}/models/{fm_name}/tokenizer")
tokenizer.save_pretrained(f'{model_path}');


