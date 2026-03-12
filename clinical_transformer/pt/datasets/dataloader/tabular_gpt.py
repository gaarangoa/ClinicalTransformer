import torch
from torch.utils.data import Dataset, DataLoader

class GPTNextTokenDataset(Dataset):
    def __init__(self, data, labels=[], **kwargs):
        self.data = data
        self.labels = labels
        self.context_window = kwargs.get('context_window', None)
        self.return_values = kwargs.get('return_values', True)
        self.return_cls = kwargs.get('return_cls', False)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_pos = len(self.data[idx][0]) - self.context_window + 0
        pix = torch.randint(0, max(1, max_pos), (1,) ).item()
        
        tokens = torch.tensor(self.data[idx][0][pix:pix+self.context_window+1])
        if self.return_values:
            values = torch.tensor(self.data[idx][1][pix:pix+self.context_window+1])
        else:
            values = []

        if tokens.shape[0] < self.context_window+1:
            pad_size = self.context_window - tokens.shape[0] + 1
            tokens = torch.nn.functional.pad(tokens, pad=[0, pad_size], mode='constant', value=0)
            if self.return_values:
                values = torch.nn.functional.pad(values, pad=[0, pad_size], mode='constant', value=0)

        if self.return_cls:
            tokens = torch.cat([torch.tensor([2]), tokens])
            if self.return_values:
                values = torch.cat([torch.tensor([1.0]), values])

        return tokens, values, idx