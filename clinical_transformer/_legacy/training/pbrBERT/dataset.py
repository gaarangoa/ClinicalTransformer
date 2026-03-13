import torch
from torch.utils.data import Dataset, DataLoader

class MaskedTokenDataset(Dataset):
    def __init__(self, tokens, values, labels=[], **kwargs):
        """
        Dataset for masked token prediction with gene expression data.
        
        Args:
            tokens: List of token sequences (one per sample)
            values: List of value sequences (one per sample, corresponding to tokens)
            labels: Optional labels (not used in current implementation)
            **kwargs: Additional parameters
        """
        self.tokens = tokens
        self.values = values
        self.labels = labels
        self.context_window = kwargs.get('context_window', None)
        self.return_values = kwargs.get('return_values', True)
        self.return_cls = kwargs.get('return_cls', False)
        self.masking_fraction = kwargs.get('masking_fraction', 0.15)
        
        # Validate input lengths
        if len(tokens) != len(values):
            raise ValueError("tokens and values must have the same length")
    
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # Get tokens and values for this sample
        sample_tokens = self.tokens[idx]
        sample_values = self.values[idx] if self.return_values else []
        
        # Determine target sequence length (before adding CLS if needed)
        target_length = self.context_window if self.return_cls else self.context_window
        
        # Calculate context window sampling
        max_pos = len(sample_tokens) - target_length + 1
        pix = torch.randint(0, max(1, max_pos), (1,) ).item()
        
        # Extract context window
        tokens = torch.tensor(sample_tokens[pix:pix+target_length], dtype=torch.long)
        if self.return_values:
            values = torch.tensor(sample_values[pix:pix+target_length])
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

        # Pad to target length if needed
        if tokens.shape[0] < target_length:
            pad_size = target_length - tokens.shape[0]
            tokens = torch.nn.functional.pad(tokens, pad=[0, pad_size], mode='constant', value=0)
            original_tokens = torch.nn.functional.pad(original_tokens, pad=[0, pad_size], mode='constant', value=0)
            if self.return_values:
                values = torch.nn.functional.pad(values, pad=[0, pad_size], mode='constant', value=0)

        # Add CLS token at the beginning if requested
        if self.return_cls:
            tokens = torch.cat([torch.tensor([2], dtype=torch.long), tokens])
            original_tokens = torch.cat([torch.tensor([2], dtype=torch.long), original_tokens])
            if self.return_values:
                values = torch.cat([torch.tensor([1.0]), values])

        return {"tokens": tokens, "values": values, "original_tokens": original_tokens}