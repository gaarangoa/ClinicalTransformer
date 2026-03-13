import torch
from torch import nn
import yaml
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import lightning as L

class BaseClinicalTransformer(nn.Transformer):
    def __init__(self, **kwargs):
        """
        
        Parameters: 
        n_modality_feature: Indicates that the feature has 1 dimension (one value). But, it can contain multiple modalities (values) per features. 
                            Therefore it can dinamically add more prior values of the features directly on here. This can be 
                            prior knowledge and other stuff. 
        
        """

        ntoken = kwargs.get('ntoken', 1024)
        ninp = kwargs.get('ninp', 128)
        nhead = kwargs.get('nhead', 2)
        nhid = kwargs.get('nhid', 256)
        nlayers = kwargs.get('nlayers', 2)
        batch_first=kwargs.get('batch_first', True)
        dropout = kwargs.get('dropout', 0.1)
        outdir = kwargs.get('output_dir', None)
        
        
        super(BaseClinicalTransformer, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, batch_first=batch_first, dropout=dropout)
        
        self.padding_mask = None
        self.n_modality_feature = kwargs.get('n_modality_feature', 1)
        self.n_classes = kwargs.get('nclasses', ntoken)
        
        self.feature_embeddings = nn.Embedding(ntoken, ninp)
        self.value_embeddings = nn.Linear(in_features=self.n_modality_feature, out_features=ninp)
        
        self.ninp = torch.tensor(ninp, dtype=torch.float32)
        self.decoder = None

    def forward(self, **kwargs):
        tokens = kwargs.get('tokens', None)
        values = kwargs.get('values', None)
        values = values.unsqueeze(-1) # add one dimension

        # padding mask
        '''
        Note: padding_mask ensures that position i is allowed to attend the unmasked positions. 
        If a BoolTensor is provided, positions with True are 
        not allowed to attend while False values will be unchanged. 
        If a FloatTensor is provided, it will be added to the attention weight.
        '''
        padding_mask = tokens == 0.0
        reversed_padding_mask = (padding_mask==False).unsqueeze(-1) # reverse true / false 
        
        token_embeddings = self.feature_embeddings(tokens)
        value_embeddings = self.value_embeddings(values)
        
        input_embeddings = torch.add(token_embeddings, value_embeddings) * torch.sqrt(self.ninp)
        output_embeddings = self.encoder(input_embeddings, src_key_padding_mask=padding_mask)

        # discard padding tokens for output embeddings
        output_embeddings = reversed_padding_mask * output_embeddings

        return output_embeddings