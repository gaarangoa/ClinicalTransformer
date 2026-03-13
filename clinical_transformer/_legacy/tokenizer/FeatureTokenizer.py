class FeatureTokenizer():
    '''
    Feature Tokenizer
    -----------------
    Simple tokenizer for a list of tokens (feature names). We added +5 to the vocabulary size, in case there is need of more special tokens.

    Input
    -----
    tokens: list with tokens

    Output
    ------
    The class generates an encoder and decoder dictionaries to map the values of the input features to integer values.

    '''
    def __init__(self, tokens):
        # self.tokens = tokens
        
        self.encoder = {i: ix+3 for ix, i in enumerate(tokens)}
        
        self.encoder['<pad>'] = 0
        self.encoder['<mask>'] = 1
        self.encoder['<cls>'] = 2
        
        self.decoder = {j:i for i,j in self.encoder.items()}
        # self.vocabulary_size = len(self.encoder) + 3 # In case we need any special tokens