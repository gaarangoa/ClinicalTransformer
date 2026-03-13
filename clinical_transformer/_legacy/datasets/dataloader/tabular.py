import torch
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, data, labels=[], **kwargs):
        self.data = data
        self.labels = labels
        self.context_window = kwargs.get('context_window')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        tokens = torch.tensor([2] + self.data[idx][0])
        values = torch.tensor([1.0] + self.data[idx][1])
        
        tokens_len = tokens.shape[0] - 1

        if tokens_len > self.context_window:
            index = torch.randperm(self.context_window) + 1
            index = torch.cat([torch.tensor([0]), index])
            tokens = tokens[index]
            values = values[index]
        else:
            pad_size = self.context_window - tokens_len
            tokens = torch.nn.functional.pad(tokens, pad=[0, pad_size], mode='constant', value=0) 
            values = torch.nn.functional.pad(values, pad=[0, pad_size], mode='constant', value=0.)

        if len(self.labels) == 0:
            label = torch.tensor(0)
        else:
            label = torch.tensor(self.labels[idx])
        
        return tokens, values, label

class TabularMaskedDataset(Dataset):
    def __init__(self, data, labels=[], **kwargs):
        self.data = data
        self.labels = labels
        self.context_window = kwargs.get('context_window')
        self.masking_fraction = kwargs.get('masking_fraction')
        self.masking_length = round(self.masking_fraction * self.context_window)
        self.mask_values = kwargs.get('mask_values', False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        tokens = torch.tensor([2] + self.data[idx][0])
        values = torch.tensor([1.0] + self.data[idx][1])
        
        tokens_len = tokens.shape[0] - 1

        # Get all available tokens and adjust to context window
        if tokens_len > self.context_window:
            index = torch.randperm(self.context_window) + 1
            index = torch.cat([torch.tensor([0]), index])
            tokens = tokens[index]
            values = values[index]
        # Add padding tokens at the end of the sequence
        else:
            pad_size = self.context_window - tokens_len
            tokens = torch.nn.functional.pad(tokens, pad=[0, pad_size], mode='constant', value=0) 
            values = torch.nn.functional.pad(values, pad=[0, pad_size], mode='constant', value=0.)  

        token_label = tokens.clone()
        value_label = values.clone()

        # add masking tokens relative to size of actual tokens
        if tokens_len <= self.context_window:
            self.masking_length = round(self.masking_fraction * tokens_len)
            mask_index = (torch.randperm(tokens_len) + 1)[:self.masking_length]
        else:
            self.masking_length = round(self.masking_fraction * self.context_window)
            mask_index = (torch.randperm(self.context_window) + 1)[:self.masking_length]

        # adds <mask> token index 1
        tokens[mask_index] = 1

        if self.mask_values: 
            values[mask_index] = 1.0

        return tokens, values, [token_label, value_label]
