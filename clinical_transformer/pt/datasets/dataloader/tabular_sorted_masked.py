import torch
from torch.utils.data import Dataset, DataLoader

class MaskedTokenDataset(Dataset):
    def __init__(self, data, labels=[], **kwargs):
        self.data = data
        self.labels = labels
        self.context_window = kwargs.get('context_window', None)
        self.return_values = kwargs.get('return_values', True)
        self.return_cls = kwargs.get('return_cls', False)
        self.masking_fraction = kwargs.get('masking_fraction', 0.15)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_pos = len(self.data[idx][0]) - self.context_window + 0
        pix = torch.randint(0, max(1, max_pos), (1,) ).item()
        
        tokens = torch.tensor(self.data[idx][0][pix:pix+self.context_window+1], dtype=torch.long)
        if self.return_values:
            values = torch.tensor(self.data[idx][1][pix:pix+self.context_window+1])
        else:
            values = []

        # Store original tokens for labels
        original_tokens = tokens.clone()
        
        # Apply masking before padding
        if self.masking_fraction > 0:
            # Only mask non-padding tokens (assuming 0 is padding)
            non_pad_mask = tokens != 0
            non_pad_indices = torch.where(non_pad_mask)[0]
            
            if len(non_pad_indices) > 0:
                # Calculate number of tokens to mask
                num_to_mask = int(len(non_pad_indices) * self.masking_fraction)
                if num_to_mask > 0:
                    # Randomly select indices to mask
                    mask_indices = non_pad_indices[torch.randperm(len(non_pad_indices))[:num_to_mask]]
                    # Replace selected tokens with mask token (1)
                    tokens[mask_indices] = 1

        if tokens.shape[0] < self.context_window+1:
            pad_size = self.context_window - tokens.shape[0] + 1
            tokens = torch.nn.functional.pad(tokens, pad=[0, pad_size], mode='constant', value=0)
            original_tokens = torch.nn.functional.pad(original_tokens, pad=[0, pad_size], mode='constant', value=0)
            if self.return_values:
                values = torch.nn.functional.pad(values, pad=[0, pad_size], mode='constant', value=0)

        if self.return_cls:
            tokens = torch.cat([torch.tensor([2], dtype=torch.long), tokens])
            original_tokens = torch.cat([torch.tensor([2], dtype=torch.long), original_tokens])
            if self.return_values:
                values = torch.cat([torch.tensor([1.0]), values])

        return {"tokens": tokens, "values": values, "original_tokens": original_tokens}