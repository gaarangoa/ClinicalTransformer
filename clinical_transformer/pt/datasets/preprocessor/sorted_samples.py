from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np 
import logging
import joblib
import yaml
import os

logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class RNAExpressionGeneOrderer(TransformerMixin, BaseEstimator):
    def __init__(self, **kwargs):
        self.categorical_features = kwargs.get('categorical_features', [])
        self.numerical_features = kwargs.get('numerical_features', [])
        self.output_dir = kwargs.get('output_dir', './out/')
        self.special_features = ['<pad>', '<mask>', '<cls>']
    
    def load(self, config_file):
        logger.info(f'Loading preprocessor from {config_file} ...')
        objects = yaml.safe_load(open(config_file, 'r'))
        for key, value in objects.items():
            setattr(self, key, value)
        
        return self

    def save(self, ):
        objects = {
            'categorical_features': self.categorical_features, 
            'numerical_features': self.numerical_features,
            'output_dir': self.output_dir,
            'feature_encoder': self.feature_encoder,
            'feature_decoder': self.feature_decoder,
            'features': self.features,
        }

        logger.info(f'Saving preprocessor, rewriting it if exists: to {self.output_dir}/preprocessor.yaml ...')
        os.makedirs(self.output_dir, exist_ok=True)
        yaml.dump(objects, open(f'{self.output_dir}/preprocessor.yaml', 'w'))

    def feature_transformer(self, tokens):
        self.feature_encoder = {i: ix+3 for ix, i in enumerate(tokens)}
        
        self.feature_encoder['<pad>'] = 0
        self.feature_encoder['<mask>'] = 1
        self.feature_encoder['<cls>'] = 2
        
        self.feature_decoder = {j:i for i,j in self.feature_encoder.items()}

    def fit(self, X, y=None, **kwargs):
        logger.info('Preprocessor: Fit')
        self.features = {i:{'type': 'cat', 'min': np.inf, 'max': -np.inf, 'encode': set()} for i in self.categorical_features}
        self.features.update({i:{'type': 'num', 'min': np.inf, 'max': -np.inf } for i in self.numerical_features})

        # add the special features here
        self.features.update({i:{'type': 'num', 'min': 0., 'max': 1. } for i in self.special_features})
        self.feature_transformer(self.categorical_features + self.numerical_features)

        