from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np 
import logging
import joblib
import yaml
import os
from tqdm import tqdm

logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Preprocessor(TransformerMixin, BaseEstimator):
    '''
    This preprocessor will take a dataset and transform the feature names to numbers (feature encoder)
    and their corresponding values to categories, these categories will be then transformed into numbers. 
    
    Next, it will apply min-max scaling to all the data (numerical and transformed categorical) and return 
    a new data version with encoded features.
    
    '''
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

        for sample in X:
            for feature, value in sample.items(): 
                try:
                    if np.isnan(value): 
                        # Discard all nan values in the input feature table
                        continue
                except:
                    pass
                
                try:
                    # if the feature is not in the listed features, discard!
                    self.features[feature];
                except:
                    continue

                feature_type = self.features.get(feature, {'type': None})['type']
                if feature_type == 'cat':
                    self.features[feature]['encode'].add(value)
                elif feature_type == 'num':
                    mini = self.features[feature].get('min', np.inf)
                    maxi = self.features[feature].get('max', -np.inf)
                    
                    self.features[feature]['min'] = mini if mini < value else value
                    self.features[feature]['max'] = maxi if maxi > value else value
        
        for feature in self.categorical_features:
            categories = self.features[feature]['encode']
            self.features[feature]['encode'] = {i:ix for ix, i in enumerate(categories)}
            self.features[feature]['decode'] = {ix:i for ix, i in enumerate(categories)}
            self.features[feature]['min'] = 0.
            self.features[feature]['max'] = len(categories)-1.
        
        self.save()
        return self
    
    def transform(self, X, y=None, **kwargs):
        # context_window = kwargs.get('context_window', None) # maximum number of features
        # logger.info(f'maximum number of features (context window) to be included in the dataset: {context_window}')

        # if context_window == None:
        #     raise ValueError(f'Context window cannot be {context_window}')

        logger.info('Preprocessor: Transform')
        newX = []
        for sample in X:
            new_sample = []
            tokens = [] # <cls> token
            values = [] # <cls> token value

            for feature, value in sample.items():
                try:
                    if np.isnan(float(value)):
                        # Discard all nan values in the input feature table
                        logger.debug('Feature: "{}"\tis NaN - ignored'.format(feature))
                        continue
                except:
                    pass

                try:
                    feature_type = self.features[feature]['type']
                except:
                    # logger.debug('Feature: "{}"\tnot present in training data - ignored'.format(feature))
                    continue

                # Skip if feature was in model but values were inf/-inf
                if self.features[feature]['max'] == -np.inf:
                    logger.debug('Feature: "{}"\tmax_value {} not present in feature tokenizer - ignored'.format(feature, self.features[feature]['max']))
                    continue  

                # if variable is categorical, we get the numerical value of it
                if feature_type == 'cat':
                    # TODO: what if the value is not present?
                    try:
                        value = self.features[feature]['encode'][value]
                    except:
                        logger.debug('Feature: "{}"\tvalue {} not present in feature tokenizer - ignored'.format(feature, value))
                        continue

                # Then we min max scale the variable (cat and num)
                numerator = value - self.features[feature]['min']
                denominator = self.features[feature]['max'] - self.features[feature]['min']

                
                
                if denominator == 0:
                    # this is a very special case where the denominator is 0, only when the feature has only 1 unique value in all the dataset. This feature should not be discarded
                    tokens.append(self.feature_encoder[feature])
                    values.append(1.0)
                else:
                    ratio = numerator / denominator
                    tokens.append(self.feature_encoder[feature])
                    values.append(max([0., ratio])) # if a value is negative then it is discarded.
                    # This is a min max normalization of log2 (TPM+1), in principle there are no negative values as TPM is always positive.
            
            # The third element tells if the input size is larger than the context window
            # add pads to end of sequences

            # pads = context_window - len(tokens)
            # replicates = 0 if pads < 0 else pads 

            newX.append([
                tokens, #+ [0]*replicates, 
                values, #+ [0.]*replicates, 
                None, # leave it for compatibility
                None # leave it for compatibility
            ])
        
        return newX

    def transform_from_ad(self, adata, **kwargs):
        """
        Transform data from an AnnData object to the CT format.
    
        Args:
            adata: AnnData object with samples in .obs and features in .var.
        Returns:
            newX: List of [tokens, values, None, None] for each sample.
        """
        logger.info('Preprocessor: Transform from AnnData')
        newX = []
        
        # Handle sparse matrices
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = np.array(adata.X)
        
        feature_names = list(adata.var_names)
        
        for i, row in tqdm(enumerate(X), total=X.shape[0]):
            tokens = []
            values = []
            
            for j, value in enumerate(row):
                feature = feature_names[j]
                
                # Skip if feature not in trained features
                if feature not in self.features:
                    logger.debug('Feature: "{}"\tnot present in training data - ignored'.format(feature))
                    continue

                # Skip if feature was in model but values were inf/-inf
                if self.features[feature]['max'] == -np.inf:
                    logger.debug('Feature: "{}"\tmax_value {} not present in feature tokenizer - ignored'.format(feature, self.features[feature]['max']))
                    continue
                    
                feature_type = self.features[feature]['type']
                
                # Handle numerical features
                if feature_type == 'num':
                    try:
                        numeric_value = float(value)
                        if np.isnan(numeric_value):
                            logger.debug('Feature: "{}"\tis NaN - ignored'.format(feature))
                            continue
                        # Ignore numerical variables with value zero
                        if numeric_value == 0.0:
                            # logger.debug('Feature: "{}"\tis zero - ignored'.format(feature))
                            continue
                        value = numeric_value
                    except (ValueError, TypeError):
                        logger.debug('Feature: "{}"\tcannot be converted to float - ignored'.format(feature))
                        continue
                
                # Handle categorical features
                elif feature_type == 'cat':
                    try:
                        value = self.features[feature]['encode'][value]
                    except KeyError:
                        logger.debug('Feature: "{}"\tvalue {} not present in feature tokenizer - ignored'.format(feature, value))
                        continue
                
                # Min-max scaling
                numerator = value - self.features[feature]['min']
                denominator = self.features[feature]['max'] - self.features[feature]['min']
                
                if denominator == 0:
                    tokens.append(self.feature_encoder[feature])
                    values.append(1.0)
                else:
                    ratio = numerator / denominator
                    tokens.append(self.feature_encoder[feature])
                    values.append(max([0., ratio]))
            
            # Maintain consistency with transform method
            newX.append([
                tokens,
                values,
            ])
        
        return newX

    def from_pretrained(path):
        return Preprocessor().load(config_file=f'{path}/tokenizer.yaml')
    

