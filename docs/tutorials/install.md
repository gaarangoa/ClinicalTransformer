# Installation
To install the clinical transformer package, use the following command:
```bash
pip install git+https://github.com/<your-org>/clinical_transformer.git
```

## Local
Clinical transformer is build on top of pytorch with the following libraries:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install lightning==2.5.0
pip install deepspeed==0.16.
```

Additional libraries used to train the clinical transformer are on ```./environments/requirements.txt``` 
```bash
pip install -r ./environments/requirements.txt
```
## Docker 
If you use it through docker you can use the environment ```ADS_CT_v2.6.0_public```.
