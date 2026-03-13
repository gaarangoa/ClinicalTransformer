# Installation

## From Git

```bash
pip install git+https://github.com/<your-org>/clinical_transformer.git
```

## Core Dependencies

Clinical Transformer is built on PyTorch with the following core libraries:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install lightning==2.5.0
pip install deepspeed==0.16.2
```

Additional libraries used for data handling and training:

```bash
pip install anndata scipy openpyxl pandas scikit-learn
```

Or install everything from the requirements file:

```bash
pip install -r ./environments/requirements.txt
```

## Docker

If you use Docker, the pre-built environment is available as `ADS_CT_v2.6.0_public`.
