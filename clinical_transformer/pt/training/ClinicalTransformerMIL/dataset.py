import torch 
from torch.utils.data import Dataset

class MILDataset(Dataset):
    def __init__(self, data, max_length=10000, random_mask_size=8000):
        self.data = data
        self.max_len = max_length
        self.random_mask_size = random_mask_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        mask: Tensor of shape (batch_size, num_instances), 1 for valid instances, 0 for padding
        '''
        x = self.data[idx]['embeddings']  # shape: (N, D)
        N, D = x.size()

        if N >= self.max_len:
            index = torch.randperm(N)[:self.max_len]
            x = x[index, :]
            mask = torch.ones(self.max_len, dtype=torch.int, device=x.device)
        else:
            pad_len = self.max_len - N
            padding = torch.zeros((pad_len, D), dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=0)

            mask = torch.cat([
                torch.ones(N, dtype=torch.int, device=x.device),   # real
                torch.zeros(pad_len, dtype=torch.int, device=x.device)  # pad
            ], dim=0)

        # Random mask (independent of pad mask)
        random_mask = torch.zeros(self.max_len, dtype=torch.int, device=x.device)
        rand_idx = torch.randperm(self.max_len)[:self.random_mask_size]
        random_mask[rand_idx] = 1
        
        return x, mask, random_mask