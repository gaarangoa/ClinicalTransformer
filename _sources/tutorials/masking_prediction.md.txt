# Masked prediction learning
This file contains a tutorial on masking prediction in transformers.

Masking prediction is a technique used in transformer models where certain tokens in the input sample are replaced with a special [MASK] token.
The model is then trained to predict the original token and its value of the masked tokens based on the context provided by the surrounding tokens.
This approach helps the model to learn bidirectional representations of the input data, capturing dependencies in both forward and backward directions.
Masking prediction is commonly used in pre-training tasks for models like BERT (Bidirectional Encoder Representations from Transformers) and the clinical transformer.

**Experiments (suggested) structure**

We suggest to have the following directory structure for running the experiments:
```bash
project_id
---| code # hosts only code
---| data # only save data (this should be ignored in any git repository)
---| models # save trained models
---| config # any custom configuration for the experiment.
```

## Setup dataset
We need to preprocess the dataset by formatting it to a list of dictionaries and save it as ```json``` format.

### Formatting the dataset
This script reads a CSV file into a pandas DataFrame, transforms the DataFrame into a JSON format, and saves the transformed data into a JSON file.
The transformation involves converting the DataFrame such that each row is represented as a dictionary with non-NaN values only.
The resulting JSON structure is a list of these dictionaries.

Example DataFrame:
| SampleID | F1 | F2 | F3 | F4 |
|----------|----|----|----|----|
|    1     | 0.2| 1.5| NaN|  M |
|    2     | 0.5| NaN| 2230| F |
|    3     | 0.1| 5.0| 652|  M |
|    4     | NaN| NaN| 25 |  M |

Example transformed JSON:
```python
[
    {SampleID: 1, F1: 0.2, F2: 1.5, F4: "M" },
    {SampleID: 2, F1: 0.5, F3: 2230, F4: "F" },
    {SampleID: 3, F1: 0.1, F2: 5.0, F3: 652, F4: "M" },
    {SampleID: 4, F3: 25, F4: "M" },
]
```

```python
import json
data = pd.read_csv('../data/dataset.csv')

transform the data to json, this depends on the format of your data
json_data = transform_to_json(data)
 
Save the process data to a json file: 
json.dump(json_data, open('../data/training.json', 'w'))
```

Note that you don't need to include missing values or ```NaN```.

### Preprocessing the dataset
After the data is formatted we need to build the pre-processor.
This processor is based on the sklearn ```preprocessing``` pipeline which performs the following actions:
* During ```fit``` method:
  * For each numerical feature extracts the ```min``` and ```max``` values in the dataset. This is used to normalize the data.
  * For categorical features, uses ordinal encoding and stores the index.
* During ```transform``` method:
  * Each numerical and categirical encoded features are normalized to `MinMax`.

```python
from clinical_transformer.pt.datasets.preprocessor.tabular import Preprocessor

Load the training data from the JSON file
data = json.load(open('../data/training.json', 'r'))

Initialize the preprocessor with specified categorical and numerical features
preprocessor = Preprocessor(
    categorical_features=['F4'],
    numerical_features=['F1', 'F2', 'F3'],
    output_dir='../models/'
)

# Fit the preprocessor on the training data
preprocessor = preprocessor.fit(data)

# Transform the training data using the fitted preprocessor
# Note: Replace '??' with the appropriate context window size
data_processed = preprocessor.transform(data, context_window=??)

# Save the processed training data to a pickle file
pickle.dump(data_processed, open('../data/training-preprocessed.pk', 'wb'))
```

## Training
Training the clinical transformer involves optimizing the model to predict masked features based on the context provided by the available features.
For large models, it may require sharding of the parameters to efficiently utilize the available hardware resources.
We have integrated parameter sharding using ```pytorch lighning``` and ```deepspeed```.
It needs to be set up in the configuration file under ```trainer``` strategy.
This allows the model to be trained on multiple devices, distributing the computational load and memory usage.

### Setting up configuration file and training
This section provides resources and guidance for setting up the configuration file and training a model using PyTorch Lightning with DeepSpeed.
It includes useful links to the Lightning DeepSpeed documentation and the trainer options documentation.

```yaml
experiment:
    name: ExperimentName
    version: 2
    save_dir: ../models/
    seed: 0
    set_float32_matmul_precision: medium
    comments: "XYZ model!"

dataset:
    input_file: ../data/training-preprocessed.pk
    num_workers: 15
    batch_size: 32
    context_window: 2000
    masking_fraction: 0.3
    mask_values: False
    shuffle: True

model:
    vocabulary_size: 3000
    embedding_size: 1024
    hidden_layer_size: 1024
    heads: 16
    layers: 10
    enable_flash_attention: true
    dropout: 0.1

loss:
    loss_token_weight: 1.0
    loss_value_weight: 1.0

optimizer:
    name: 'DeepSpeedCPUAdam' # Optimizer to use (DeepSpeed CPU Adam in this case). options: 'torch.optim.adam|torch.optim.adamW|DeepSpeedCPUAdam'
    params:
        lr: 0.00001 # Learning rate for the optimizer.

trainer:
  epochs: 5000
  log_every_n_steps: 1
  deterministic: false
  devices: 8
  accelerator: 'cuda'
  precision: 'bf16-mixed'
  strategy: 
    name: deepspeed # Use DeepSpeed for distributed training.
    params: 
        stage: 3 # DeepSpeed optimization stage.
        offload_optimizer: True # Offload optimizer state to CPU.
        offload_parameters: True # Offload model parameters to CPU.
```
Finally, save the file as ```config.yaml```.

**Useful resources**: 
* <a href="https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html" target="_blank">Lightning deepspeed documentation</a>
* <a href="https://lightning.ai/docs/pytorch/stable/common/trainer.html" target="_blank"> trainer </a> options.

**Optimizations**: 
1. For building larger models we recommend to use ```precision: "16-mixed"```. However, it is known that mixed precision tend to be unstable, therefore setting ```accumulate_grad_batches``` to values larger than ```1``` can help to overcome this issue.

### Execute the Training

The following script sets up the environment and runs the training for masked self-supervised learning (SSL) using PyTorch's `torchrun` utility.

1. **Set PYTHONPATH**: Adds the local Python 3.11 dist-packages directory to the `PYTHONPATH` environment variable.
2. **Set PATH**: Adds the local bin directory to the `PATH` environment variable.

Make sure to adjust the paths and configuration file as needed for your specific setup.

```bash
export PYTHONPATH=$HOME/python/usr/local/lib/python3.11/dist-packages/:$PYTHONPATH
export PATH=$HOME/python/usr/local/bin:$PATH

torchrun --nproc_per_node=4 \
    --no-python \
    train_MaskedSSL \
    config.yaml
```

## Extract embeddings

In this section, we will load the trained model and use it to make predictions or extract embeddings from new data.

#### Define your model's path
```python
path = '/path/to/your/pretrained/model/'
fmname = 'ssl_rnaseq'
epoch = 8999
version = 1
```

#### converting model back to fp32 (optional)
If the model was trained with ```deep speed```, it will generate the model in several files, we need to create one single file that we can load with ```pytorch```. The model is saved as ```lightning_model.pt```.

```python
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler
torch.serialization.add_safe_globals([ZeroStageEnum, LossScaler])

convert_zero_checkpoint_to_fp32_state_dict(
    f"{path}/models/{fmname}/version_{version}/models/epoch={epoch}.ckpt",
    f"{path}/models/{fmname}/version_{version}/models/epoch={epoch}.ckpt/lightning_model.pt"
);

```

#### Example code for extracting embeddings

```python
import json
import pickle
from clinical_transformer.pt.datasets.preprocessor.tabular import Preprocessor
from clinical_transformer.pt.datasets.dataloader.tabular import TabularDataset
from clinical_transformer.pt.models.masked_prediction import MaskedSSL
from torch.utils.data import DataLoader
import torch
import os

# Load the new data
new_data = json.load(open('../data/new_data.json', 'r'))

# let's use CPU for inference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# load the model
model = MaskedSSL.load_from_checkpoint(f'{path}/models/epoch={epoch}.ckpt/lightning_model.pt').to('cpu')

# Load the preprocessor configuration
preprocessor = Preprocessor().load(config_file='../models/preprocessor.yaml')

# setup parameters for loading data. 
# Context window can be anything and different to how the model was trained.
context_window = ??
batch_size = 100
num_workers = 1

# For inference for extracting embeddings we are not using the masked preprocessor. Instead, we use a preprocessor that takes the data as it is, only padding 0's at the end if needed.  
data_processed = preprocessor.transform(new_data, context_window=context_window)
dataset = TabularDataset(data_processed, context_window=context_window)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# As we are not making predictions and to save memory space, we remove the decoder form the model and only use the encoder.
model.token_decoder = None
model.value_decoder = None

model.eval();
emb = []
for tokens, values, labels in tqdm(dataloader):
    e = model.encoder(tokens=tokens, values=values)
    emb.append(e.detach().numpy())

emb = np.concatenate(emb)
```

This code snippet demonstrates how to load the preprocessor and model, preprocess new data, make predictions, and extract embeddings. Adjust the file paths and parameters as needed for your specific setup. **Remember to properly setup the experiment to your use case and data**

## Finetuning
In this section, we provide a comprehensive guide on how to fine-tune a pre-trained model and run a downstream model. The purpose of this section is to demonstrate the process of loading a pre-trained model, preprocessing new data, and fine-tuning the model for a specific task. The steps include setting up the model path, preparing the dataset, configuring the training parameters, and running the fine-tuning process to optimize the model for the task.

Note that this is highly flexible and we could train the model to perform any downstream task, we'll highlight in here, regression, classification and survival. 

Note that we are only including the training portion and a validation / test would be needed in real cases. 

### Load the pretrained model
```python
# set pretrained model
path = '/path/to/your/pretrained/model/'
fmname = 'ssl_rnaseq'
epoch = 8999
version = 1

# load the pretrained model
device = 'cuda'
model = MaskedSSL.load_from_checkpoint(f'{path}/models/{fmname}/version_{version}/models/epoch={epoch}.ckpt/lightning_model.pt').to(device)
```

### Load and process data
```python
# we load the preprocessor so we can process new data for finetuning using the processor created during the pre-training stage.
preprocessor = Preprocessor().load(config_file=f'{path}/models/preprocessor.yaml')

# we load the data in JSON format
tcga = pickle.load(open('/path/to/your/data/RNASeq+tcga+cptac.pk', 'rb'))

# We are finetuning on a subset of the input features in this case we will use the following feature set, we use it as a dictionary for simplicity: 
features = {
    'ENSG00000073861': True,
    'ENSG00000113263': True,
    'ENSG00000167286': True,
    'ENSG00000198851': True,
    'ENSG00000160654': True,
    'ENSG00000277734': True,
    'ENSG00000211751': True,
    'ENSG00000211772': True,
    'ENSG00000178562': True,
    'ENSG00000110448': True,
    'ENSG00000163519': True,
    'CLINICAL_FEATURE_1': True,
    'CLINICAL_FEATURE_2': True
}

# Here we are selecting a small set of features to finetune
tcga_subset = [{i:k[i] for i in k if features.get(i, False)} for k in tcga]

# We get the sample ids (for some posterior analysis)
ids = np.array([i['sample_id'] for i in tcga])

# We define our labels
labels = np.array([i['label_column'] for i in tcga])
labels_dict = {k:ix for ix, k in enumerate(set(labels))}
labels = np.array([labels_dict[i] for i in labels])
```

### Setup data loader and context
This part is important because we are telling the fine tuning model what context window to use.
```python
context_window = len(features)
batch_size = 2024
num_workers = 1

tcga_data_processed = preprocessor.transform(tcga_subset, context_window=context_window)
tcga_dataset = TabularDataset(tcga_data_processed, labels, context_window=context_window, masking_fraction=None)
tcga_dataloader = DataLoader(tcga_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
```

### Classification
Here we can remove the parts of the pre-trained model that we are not going to use, such as the decoder of the clinical transformer, this will make our fine tuning model smaller. 
```python
del model.token_decoder
del model.value_decoder
del model.loss
del model.encoder.decoder

model.decoder = torch.nn.Linear(1024, len(labels_dict), bias=True)
model.to(device)
model

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

for e in range(20):
    running_loss = 0
    model.train()
    for step, (t, v, labels)  in enumerate(tqdm(tcga_dataloader)):
        t, v, labels = t.to(device), v.to(device), labels.to(device)

        optimizer.zero_grad()
        cls_embeddings = model.encoder(tokens=t, values=v)[:, 0, :]
        ypred = model.decoder(cls_embeddings)

        loss = criterion(ypred, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'epoch: {e}, loss: {((running_loss + 1) / (step + 1)) }')
```

