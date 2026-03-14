import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
from scipy import sparse
import anndata as ad
import os
import pickle


def collate_variable_length(batch):
    """Collate variable-length samples by padding to max length in the batch.

    Pads tokens with 0 (padding token), values with 0.0, labels with 0.0.
    The model's padding_mask (tokens != 0) handles these automatically.
    When all sequences are the same length this is a no-op (just stacks).
    """
    tokens = pad_sequence([s['tokens'] for s in batch], batch_first=True, padding_value=0)
    values = pad_sequence([s['values'] for s in batch], batch_first=True, padding_value=0.0)
    labels = pad_sequence([s['labels'] for s in batch], batch_first=True, padding_value=0.0)
    return {'tokens': tokens, 'values': values, 'labels': labels}


class MaskedTokenDataset(Dataset):
    """
    PyTorch Dataset for masked token prediction with gene expression data.
    
    This dataset implements the masked language modeling objective specifically
    designed for continuous gene expression values in vnBERT. Unlike
    traditional masked language modeling that uses discrete tokens, this
    dataset handles continuous numerical values representing gene expression
    levels.
    
    Key Features:
    - Context window sampling: Randomly selects a subset of genes per sample
    - End-of-sequence masking: Masks tokens from the end rather than randomly
    - Value preservation: Maintains original expression values as labels
    - CLS token integration: Automatically adds classification token
    
    Args:
        tokens (List[List[int]]): List of tokenized samples, where each sample
            is a list of token IDs corresponding to genes
        values (List[List[float]]): List of expression value samples, where
            each sample contains normalized expression values for corresponding
            genes
        context_window (int, optional): Maximum number of genes to sample per
            training example. If None, uses all available genes. Recommended
            values: 1024-4096 depending on available memory
        mask_prob (float): Probability of masking tokens for prediction.
            Standard BERT uses 0.15, but can be adjusted for genomic data
        **kwargs: Additional arguments for future extensibility
    
    Raises:
        ValueError: If input_ids and raw_values have different lengths
    
    Returns:
        Dict containing:
        - tokens (torch.LongTensor): Gene token IDs with CLS token prepended
        - values (torch.FloatTensor): Expression values with masked positions
            set to -10.0 and CLS value set to 1.0
        - labels (torch.FloatTensor): Original expression values for loss
            computation, with CLS label set to 1.0
    
    Example:
        >>> dataset = MaskedTokenDataset(
        ...     tokens=encoded['input_ids'],
        ...     values=encoded['robust_zscore_values'],
        ...     context_window=2048,
        ...     mask_prob=0.15
        ... )
        >>> sample = dataset[0]
        >>> print(f"Tokens shape: {sample['tokens'].shape}")
        >>> print(f"Values shape: {sample['values'].shape}")
        >>> print(f"Labels shape: {sample['labels'].shape}")
    """
    def __init__(self, tokens, values, context_window=None, mask_prob=0.15,
                 **kwargs):
        """
        Dataset for masked token prediction with gene expression data.

        Args:
            tokens: List/array of tokenized samples
            values: List/array of expression value samples
            context_window: Number of genes to randomly select per sample
                           (None = use all)
            mask_prob: Probability of masking each token (default 0.15)
        """
        if len(tokens) != len(values):
            raise ValueError("input_ids and raw_values must have same length")

        # Pre-convert to tensors once at init to avoid per-sample conversion
        self.input_ids = [torch.as_tensor(t, dtype=torch.long) for t in tokens]
        self.raw_values = [torch.as_tensor(v, dtype=torch.float32) for v in values]
        self.context_window = context_window
        self.mask_prob = mask_prob
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Retrieve and process a single training sample.
        
        This method implements the core data processing pipeline for vnBERT
        training, including context window sampling, masking strategy, and
        CLS token addition.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Dict[str, torch.Tensor]: Processed sample containing:
                - tokens: Gene token IDs with CLS token (LongTensor)
                - values: Expression values with masking applied (FloatTensor)
                - labels: Original expression values for loss computation
        
        Processing Steps:
        1. Extract tokens and values for the specified sample
        2. Convert to PyTorch tensors with appropriate dtypes
        3. Apply context window sampling if specified
        4. Store original values as prediction labels
        5. Apply end-of-sequence masking strategy
        6. Prepend CLS token to sequence
        """
        tokens = self.input_ids[idx]
        values = self.raw_values[idx]

        # Context window sampling
        if (self.context_window is not None and
                len(tokens) > self.context_window):
            indices = torch.randperm(len(tokens))[:self.context_window]
            tokens = tokens[indices]
            values = values[indices]

        n = len(tokens)
        seq_len = n + 1  # +1 for CLS
        num_masked = int(n * self.mask_prob) if self.mask_prob > 0 else 0

        # Pre-allocate output tensors (CLS + sequence)
        out_tokens = torch.empty(seq_len, dtype=torch.long)
        out_tokens[0] = 2
        out_tokens[1:] = tokens

        out_labels = torch.empty(seq_len, dtype=torch.float32)
        out_labels[0] = 1.0
        out_labels[1:] = values

        out_values = out_labels.clone()
        if num_masked > 0:
            out_values[-num_masked:] = -10.0

        return {
            "tokens": out_tokens,
            "values": out_values,
            "labels": out_labels
        }


class MaskedTokenDatasetFromPytorchObject(Dataset):
    """
    PyTorch Dataset for masked token prediction with sparse PyTorch gene expression matrices.

    This dataset extends the masked language modeling objective to work with sparse
    matrices from PyTorch tensors. It automatically filters out zero values and uses
    the column indices as token IDs, making it memory-efficient for large genomic
    datasets with many zero expression values.
    
    Key Features:
    - Sparse matrix support: Efficiently handles large sparse expression matrices
    - Automatic zero filtering: Removes zero values and uses non-zero indices as tokens
    - Context window sampling: Randomly selects subset of non-zero genes per sample
    - End-of-sequence masking: Applies same masking strategy as MaskedTokenDataset
    - Memory efficient: Only processes non-zero expression values
    
    Args:
        values: Shared PyTorch tensor (samples x genes)
            Expected format: PyTorch tensor where rows are samples
            and columns correspond to gene token IDs. Can be sparse or dense tensor.
        context_window (int, optional): Maximum number of genes to sample per
            training example from non-zero values. If None, uses all non-zero genes
        mask_prob (float): Probability of masking tokens for prediction.
            Default 0.15 following BERT standard
        **kwargs: Additional arguments for future extensibility
    
    Raises:
        ValueError: If values is not a PyTorch tensor
        IndexError: If sample index is out of bounds
    
    Returns:
        Dict containing:
        - tokens (torch.LongTensor): Non-zero gene token IDs (column indices) with CLS
        - values (torch.FloatTensor): Non-zero expression values with masking applied
        - labels (torch.FloatTensor): Original non-zero expression values for loss
    
    Example:
        >>> # Using a shared PyTorch tensor
        >>> values_tensor = torch.tensor(expression_data)  # Shape: (samples, genes)
        >>> dataset = MaskedTokenDatasetFromH5Ann(
        ...     values=values_tensor,
        ...     context_window=2048,
        ...     mask_prob=0.15
        ... )
        >>> sample = dataset[0]
        >>> print(f"Non-zero tokens: {sample['tokens'].shape}")
        >>> print(f"Expression values: {sample['values'].shape}")
    """
    
    def __init__(self, values, context_window=None, mask_prob=0.15, **kwargs):
        """
        Initialize dataset for shared PyTorch tensor processing.
        
        Args:
            values: Shared PyTorch tensor (samples x genes)
            context_window: Max number of non-zero genes per sample (None = all)
            mask_prob: Probability of masking each token (default 0.15)
        """
        self.values = values
        self.context_window = context_window
        self.mask_prob = mask_prob
        
        # Validate input is PyTorch tensor
        if not isinstance(values, torch.Tensor):
            raise ValueError("values must be a PyTorch tensor")
        
        # Store number of samples
        self.n_samples = values.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Retrieve and process a single sample from shared PyTorch tensor.
        
        This method extracts non-zero values from the tensor row,
        uses column indices as token IDs, and applies the same masking
        strategy as the original MaskedTokenDataset.
        
        Args:
            idx (int): Sample index (row) in the tensor
            
        Returns:
            Dict[str, torch.Tensor]: Processed sample containing:
                - tokens: Non-zero gene token IDs with CLS token
                - values: Expression values with masking applied
                - labels: Original expression values for loss computation
        
        Processing Steps:
        1. Extract tensor row 
        2. Find non-zero positions and values
        3. Use column indices as token IDs
        4. Apply context window sampling on non-zero elements
        5. Apply masking strategy and add CLS token
        """
        # Extract the row for this sample
        sample_row = self.values[idx, :]
        
        # Handle sparse PyTorch tensors
        if sample_row.is_sparse:
            # Convert sparse tensor to COO format
            sample_coo = sample_row.coalesce()
            # Extract non-zero column indices as token IDs
            sample_tokens = sample_coo.indices()[0].tolist()
            # Extract non-zero values
            sample_values = sample_coo.values().tolist()
        else:
            # Handle dense PyTorch tensor
            # Find non-zero positions
            nonzero_mask = sample_row != 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]
            sample_tokens = nonzero_indices.tolist()
            sample_values = sample_row[nonzero_indices].tolist()
        
        # Skip if no non-zero values found
        if len(sample_tokens) == 0:
            # Return empty sample with just CLS token
            return {
                "tokens": torch.tensor([2], dtype=torch.long),  # CLS token
                "values": torch.tensor([1.0], dtype=torch.float32),
                "labels": torch.tensor([1.0], dtype=torch.float32)
            }
        
        # Convert to tensors
        tokens = torch.tensor(sample_tokens, dtype=torch.long)
        values = torch.tensor(sample_values, dtype=torch.float32)
        
        # Apply context window sampling if specified
        # Randomly select from available non-zero tokens
        if (self.context_window is not None and 
                len(tokens) > self.context_window):
            # Randomly select indices without replacement
            indices = torch.randperm(len(tokens))[:self.context_window]
            tokens = tokens[indices]
            values = values[indices]
        
        # Store original values as labels for loss computation
        labels = values.clone()
        
        # Apply masking strategy: mask tokens from the end of sequence
        # Same strategy as original MaskedTokenDataset
        if self.mask_prob > 0:
            num_tokens_to_mask = int(len(tokens) * self.mask_prob)
            if num_tokens_to_mask > 0:
                # Set masked positions to special value (-10.0)
                values[-num_tokens_to_mask:] = -10.0
        
        # Add CLS token at the beginning (token ID 2)
        tokens = torch.cat([torch.tensor([2], dtype=torch.long), tokens])
        values = torch.cat([torch.tensor([1.0], dtype=torch.float32), values])
        labels = torch.cat([torch.tensor([1.0], dtype=torch.float32), labels])
        
        return {
            "tokens": tokens,
            "values": values,
            "labels": labels
        }


class MaskedTokenDatasetFromAnnData(Dataset):
    """
    PyTorch Dataset for masked token prediction with AnnData files read on-demand from disk.
    
    This dataset provides memory-efficient access to large AnnData files by reading
    samples on-demand using memory mapping (r+ mode). It automatically filters out
    zero values and uses column indices as token IDs, making it suitable for very
    large genomic datasets that don't fit in memory.
    
    Key Features:
    - Memory-mapped file access: Reads samples on-demand from disk using r+ mode
    - Sparse matrix support: Efficiently handles sparse expression matrices
    - Automatic zero filtering: Removes zero values and uses non-zero indices as tokens
    - Context window sampling: Randomly selects subset of non-zero genes per sample
    - End-of-sequence masking: Applies same masking strategy as MaskedTokenDataset
    - Memory efficient: Only loads requested samples, not the entire dataset
    
    Args:
        anndata_path (str): Path to the AnnData h5ad file on disk
        context_window (int, optional): Maximum number of genes to sample per
            training example from non-zero values. If None, uses all non-zero genes
        mask_prob (float): Probability of masking tokens for prediction.
            Default 0.15 following BERT standard
        **kwargs: Additional arguments for future extensibility
    
    Raises:
        FileNotFoundError: If the AnnData file doesn't exist
        ValueError: If the file cannot be read as AnnData
        IndexError: If sample index is out of bounds
    
    Returns:
        Dict containing:
        - tokens (torch.LongTensor): Non-zero gene token IDs (column indices) with CLS
        - values (torch.FloatTensor): Non-zero expression values with masking applied
        - labels (torch.FloatTensor): Original non-zero expression values for loss
    
    Example:
        >>> # Using AnnData file on disk
        >>> dataset = MaskedTokenDatasetFromAnnData(
        ...     anndata_path='/path/to/expression_data.h5ad',
        ...     context_window=2048,
        ...     mask_prob=0.15
        ... )
        >>> sample = dataset[0]
        >>> print(f"Non-zero tokens: {sample['tokens'].shape}")
        >>> print(f"Expression values: {sample['values'].shape}")
    """
    
    def __init__(self, anndata_path, context_window=None, mask_prob=0.15, filter_zeros=True, **kwargs):
        """
        Initialize dataset for on-demand AnnData file reading.

        Args:
            anndata_path: Path to AnnData h5ad file on disk
            context_window: Max number of non-zero genes per sample (None = all)
            mask_prob: Probability of masking each token (default 0.15)
            filter_zeros: If True, only non-zero values are used as tokens (default).
                If False, all features are included (useful when zeros are valid categories).
        """
        self.anndata_path = anndata_path
        self.context_window = context_window
        self.mask_prob = mask_prob
        self.filter_zeros = filter_zeros
        
        # Validate file exists
        if not os.path.exists(anndata_path):
            raise FileNotFoundError(f"AnnData file not found: {anndata_path}")
        
        # Open AnnData file in read mode to get basic info
        try:
            # Use backed mode to avoid loading entire file
            self.adata = ad.read_h5ad(anndata_path, backed='r+')
            self.n_samples = self.adata.n_obs
            self.n_genes = self.adata.n_vars
        except Exception as e:
            raise ValueError(f"Cannot read AnnData file {anndata_path}: {str(e)}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Retrieve and process a single sample from AnnData file on disk.
        
        This method reads a single sample from the memory-mapped AnnData file,
        extracts non-zero values, uses column indices as token IDs, and applies
        the same masking strategy as the original MaskedTokenDataset.
        
        Args:
            idx (int): Sample index (row) in the AnnData file
            
        Returns:
            Dict[str, torch.Tensor]: Processed sample containing:
                - tokens: Non-zero gene token IDs with CLS token
                - values: Expression values with masking applied
                - labels: Original expression values for loss computation
        
        Processing Steps:
        1. Read single sample row from disk using memory mapping
        2. Find non-zero positions and values
        3. Use column indices as token IDs
        4. Apply context window sampling on non-zero elements
        5. Apply masking strategy and add CLS token
        """
        # Read single sample from disk (memory-mapped access)
        # This only loads the requested row, not the entire matrix
        sample_row = self.adata.X[idx, :]
        
        # Handle different matrix formats
        if hasattr(sample_row, 'toarray'):
            # Sparse matrix - convert to dense for processing
            sample_dense = sample_row.toarray().flatten()
        elif hasattr(sample_row, 'todense'):
            # Another sparse format
            sample_dense = np.asarray(sample_row.todense()).flatten()
        else:
            # Already dense
            sample_dense = np.asarray(sample_row).flatten()
        
        if self.filter_zeros:
            # Find non-zero positions and values
            nonzero_mask = sample_dense != 0.0
            nonzero_indices = np.where(nonzero_mask)[0]

            # Extract non-zero values and use indices as token IDs
            sample_tokens = nonzero_indices.tolist()
            sample_values = sample_dense[nonzero_indices].tolist()

            # Skip if no non-zero values found
            if len(sample_tokens) == 0:
                # Return empty sample with just CLS token
                return {
                    "tokens": torch.tensor([2], dtype=torch.long),  # CLS token
                    "values": torch.tensor([1.0], dtype=torch.float32),
                    "labels": torch.tensor([1.0], dtype=torch.float32)
                }
        else:
            # Use all features (zeros are valid values)
            sample_tokens = np.arange(len(sample_dense)).tolist()
            sample_values = sample_dense.tolist()
        
        # Convert to tensors
        tokens = torch.tensor(sample_tokens, dtype=torch.long)
        values = torch.tensor(sample_values, dtype=torch.float32)
        
        # Apply context window sampling if specified
        # Randomly select from available non-zero tokens
        if (self.context_window is not None and 
                len(tokens) > self.context_window):
            # Randomly select indices without replacement
            indices = torch.randperm(len(tokens))[:self.context_window]
            tokens = tokens[indices]
            values = values[indices]
        
        # Store original values as labels for loss computation
        labels = values.clone()
        
        # Apply masking strategy: mask tokens from the end of sequence
        # Same strategy as original MaskedTokenDataset
        if self.mask_prob > 0:
            num_tokens_to_mask = int(len(tokens) * self.mask_prob)
            if num_tokens_to_mask > 0:
                # Set masked positions to special value (-10.0)
                values[-num_tokens_to_mask:] = -10.0
        
        # Add CLS token at the beginning (token ID 2)
        tokens = torch.cat([torch.tensor([2], dtype=torch.long), tokens])
        values = torch.cat([torch.tensor([1.0], dtype=torch.float32), values])
        labels = torch.cat([torch.tensor([1.0], dtype=torch.float32), labels])
        
        return {
            "tokens": tokens,
            "values": values,
            "labels": labels
        }
    
    def __del__(self):
        """
        Cleanup method to properly close the AnnData file.
        """
        if hasattr(self, 'adata') and self.adata is not None:
            try:
                # Close the backed AnnData file
                if hasattr(self.adata, 'file') and self.adata.file is not None:
                    self.adata.file.close()
            except:
                pass  # Ignore cleanup errors


class MaskedPriorTokenDataset(Dataset):
    """
    PyTorch Dataset for masked token prediction with biologically-informed gene selection.
    
    This dataset extends the masked language modeling objective to use biological prior
    knowledge for gene selection instead of purely random sampling. It leverages
    biological process annotations to select genes that are functionally related,
    potentially improving the model's understanding of biological relationships.
    
    Key Features:
    - Biological prior-informed sampling: Uses predefined gene sets (biological processes)
    - Hierarchical sampling strategy: Select fewer processes, more genes per process
    - Memory-mapped file access: Reads samples on-demand from AnnData files
    - Context window management: Ensures consistent sequence lengths for training
    - End-of-sequence masking: Applies same masking strategy as other datasets
    
    Sampling Strategy:
    1. Randomly select a subset of biological processes from the prior list
    2. From selected processes, randomly sample genes to reach target context window
    3. Intersect with available non-zero genes for the current sample
    4. Apply masking and return formatted tensors
    
    Args:
        anndata_path (str): Path to the AnnData h5ad file on disk
        prior_path (str): Path to pickle file containing list of biological processes,
            where each process is a list of gene indices (column indices in AnnData)
        context_window (int, optional): Target number of genes per sample.
            Default 1000 to match your requirements
        n_processes (int, optional): Number of biological processes to sample.
            Default 100 for hierarchical sampling (fewer processes, more genes each)
        mask_prob (float): Probability of masking tokens for prediction.
            Default 0.15 following BERT standard
        **kwargs: Additional arguments for future extensibility
    
    Raises:
        FileNotFoundError: If the AnnData file or prior pickle file doesn't exist
        ValueError: If the file cannot be read as AnnData or prior is invalid
        IndexError: If sample index is out of bounds
    
    Returns:
        Dict containing:
        - tokens (torch.LongTensor): Biologically-selected gene token IDs with CLS
        - values (torch.FloatTensor): Expression values with masking applied
        - labels (torch.FloatTensor): Original expression values for loss
    
    Example:
        >>> # Using biological priors for gene selection from pickle file
        >>> dataset = MaskedPriorTokenDataset(
        ...     anndata_path='/path/to/expression_data.h5ad',
        ...     prior_path='/path/to/biological_processes.pkl',
        ...     context_window=1000,
        ...     n_processes=100,
        ...     mask_prob=0.15
        ... )
        >>> sample = dataset[0]
        >>> print(f"Biologically-selected tokens: {sample['tokens'].shape}")
    """
    
    def __init__(self, anndata_path, prior_path, context_window=1000, n_processes=100, 
                 mask_prob=0.15, **kwargs):
        """
        Initialize dataset with biological prior knowledge from pickle file.
        
        Args:
            anndata_path: Path to AnnData h5ad file on disk
            prior_path: Path to pickle file containing list of biological processes
            context_window: Target number of genes per sample (default 1000)
            n_processes: Number of biological processes to sample (default 100)
            mask_prob: Probability of masking each token (default 0.15)
        """
        self.anndata_path = anndata_path
        self.prior_path = prior_path
        self.context_window = context_window
        self.n_processes = n_processes
        self.mask_prob = mask_prob
        
        # Validate inputs
        if not os.path.exists(anndata_path):
            raise FileNotFoundError(f"AnnData file not found: {anndata_path}")
        
        if not os.path.exists(prior_path):
            raise FileNotFoundError(f"Prior pickle file not found: {prior_path}")
        
        # Load prior from pickle file
        try:
            with open(prior_path, 'rb') as f:
                self.prior = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Cannot load prior from pickle file {prior_path}: {str(e)}")
        
        # Validate prior structure
        if not isinstance(self.prior, list) or len(self.prior) == 0:
            raise ValueError("prior must be a non-empty list of gene index lists")
        
        if n_processes > len(self.prior):
            raise ValueError(f"n_processes ({n_processes}) cannot exceed number of available processes ({len(self.prior)})")
        
        # Open AnnData file in read mode to get basic info
        try:
            # Use backed mode to avoid loading entire file
            self.adata = ad.read_h5ad(anndata_path, backed='r+')
            self.n_samples = self.adata.n_obs
            self.n_genes = self.adata.n_vars
        except Exception as e:
            raise ValueError(f"Cannot read AnnData file {anndata_path}: {str(e)}")
        
        # Validate that gene indices in prior are within bounds
        max_gene_idx = max(max(process) for process in self.prior if len(process) > 0)
        if max_gene_idx >= self.n_genes:
            raise ValueError(f"Gene index {max_gene_idx} in prior exceeds number of genes {self.n_genes}")
        
        # Pre-compute some statistics for efficiency
        self.process_sizes = [len(process) for process in self.prior]
        self.total_unique_genes = len(set(gene for process in self.prior for gene in process))
        
        # Pre-convert all process gene lists to sets for faster intersection operations
        self.prior_sets = [set(process) for process in self.prior]
        
        # print(f"Initialized MaskedPriorTokenDataset:")
        # print(f"  - {len(self.prior)} biological processes available")
        # print(f"  - {self.total_unique_genes} unique genes across all processes")
        # print(f"  - Will sample {n_processes} processes per sample")
        # print(f"  - Target context window: {context_window} genes")
    
    def __len__(self):
        return self.n_samples
    
    def _select_genes_from_prior(self, available_genes):
        """
        Select genes using biological prior knowledge with cumulative random process selection.
        
        Implements cumulative unique gene selection strategy:
        1. Randomly shuffle the order of biological processes
        2. Iterate through processes in random order
        3. Cumulatively add unique expressed genes from each process
        4. Stop when context_window genes are reached
        5. Ensures consistent sequence length without post-selection sampling
        
        Args:
            available_genes (set): Set of gene indices that are expressed (non-zero) in current sample
        
        Returns:
            list: List of selected gene indices in the order they were added (preserves randomization)
        """
        # Pre-allocate array for maximum possible genes (faster than list append)
        selected_genes = np.empty(self.context_window, dtype=np.int64)
        seen_genes = set()
        n_selected = 0
        
        # Randomly shuffle process indices for random iteration order
        shuffled_process_indices = np.random.permutation(len(self.prior_sets))
        
        # Iterate through processes in random order
        for idx in shuffled_process_indices:
            # Find expressed genes in this process that we haven't seen yet
            expressed_genes_in_process = self.prior_sets[idx].intersection(available_genes)
            new_genes = expressed_genes_in_process - seen_genes
            
            if len(new_genes) > 0:
                # Convert to numpy array for efficient operations
                new_genes_array = np.array(list(new_genes), dtype=np.int64)
                np.random.shuffle(new_genes_array)
                
                # Calculate how many genes we can add
                remaining_slots = self.context_window - n_selected
                n_to_add = min(len(new_genes_array), remaining_slots)
                
                # Vectorized copy into pre-allocated array
                selected_genes[n_selected:n_selected + n_to_add] = new_genes_array[:n_to_add]
                
                # Update seen genes set (batch update is faster than individual adds)
                seen_genes.update(new_genes_array[:n_to_add])
                n_selected += n_to_add
                
                # Early exit if we've reached target
                if n_selected >= self.context_window:
                    return selected_genes[:n_selected].tolist()
        
        # Return whatever we collected (might be less than context_window)
        return selected_genes[:n_selected].tolist()
    
    def __getitem__(self, idx):
        """
        Retrieve and process a single sample with biologically-informed gene selection.
        
        This method combines biological prior knowledge with expression data:
        1. First gets non-zero genes from the sample 
        2. Then uses biological priors to select from expressed genes
        3. Applies standard masking and formatting
        
        Args:
            idx (int): Sample index (row) in the AnnData file
            
        Returns:
            Dict[str, torch.Tensor]: Processed sample containing:
                - tokens: Biologically-selected gene token IDs with CLS token
                - values: Expression values with masking applied
                - labels: Original expression values for loss computation
        
        Processing Steps:
        1. Read sample from disk and find non-zero genes first
        2. Use biological priors to select from available non-zero genes
        3. Apply masking strategy and add CLS token
        """
        # Step 1: Read single sample from disk (memory-mapped access)
        sample_row = self.adata.X[idx, :]
        
        # Handle different matrix formats more efficiently
        if hasattr(sample_row, 'toarray'):
            # Sparse matrix - convert to dense for processing
            sample_dense = sample_row.toarray().ravel()  # ravel is faster than flatten
        elif hasattr(sample_row, 'todense'):
            # Another sparse format
            sample_dense = np.asarray(sample_row.todense()).ravel()
        else:
            # Already dense
            sample_dense = np.asarray(sample_row).ravel()
        
        # Find non-zero positions - use nonzero() directly for speed
        nonzero_indices = np.nonzero(sample_dense)[0]
        
        # Early exit if no non-zero genes
        if len(nonzero_indices) == 0:
            return {
                "tokens": torch.tensor([2], dtype=torch.long),
                "values": torch.tensor([1.0], dtype=torch.float32),
                "labels": torch.tensor([1.0], dtype=torch.float32)
            }
        
        # Convert to set for fast intersection
        nonzero_genes = set(nonzero_indices)
        
        # Step 2: Select genes using biological priors from available non-zero genes
        # Returns list of genes in random order (randomization done during selection)
        final_genes_list = self._select_genes_from_prior(available_genes=nonzero_genes)
        
        # Early exit if no genes selected
        if len(final_genes_list) == 0:
            return {
                "tokens": torch.tensor([2], dtype=torch.long),
                "values": torch.tensor([1.0], dtype=torch.float32),
                "labels": torch.tensor([1.0], dtype=torch.float32)
            }
        
        # Convert to numpy array for efficient indexing (already in random order)
        final_gene_array = np.array(final_genes_list, dtype=np.int64)
        
        # Get expression values for selected genes (vectorized operation)
        sample_values = sample_dense[final_gene_array]
        
        # Convert to tensors (genes already in random order from selection process)
        tokens = torch.from_numpy(final_gene_array)
        values = torch.from_numpy(sample_values).float()
        
        # Store original values as labels for loss computation
        labels = values.clone()
        
        # Apply masking strategy: mask tokens from the end of sequence
        if self.mask_prob > 0:
            num_tokens_to_mask = int(len(tokens) * self.mask_prob)
            if num_tokens_to_mask > 0:
                # Set masked positions to special value (-10.0)
                values[-num_tokens_to_mask:] = -10.0
        
        # Add CLS token at the beginning (token ID 2) - use efficient concatenation
        tokens = torch.cat([torch.tensor([2], dtype=torch.long), tokens])
        values = torch.cat([torch.tensor([1.0], dtype=torch.float32), values])
        labels = torch.cat([torch.tensor([1.0], dtype=torch.float32), labels])
        
        return {
            "tokens": tokens,
            "values": values,
            "labels": labels
        }
    
    def get_stats(self):
        """
        Get statistics about the dataset and prior knowledge.
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        return {
            "n_samples": self.n_samples,
            "n_genes": self.n_genes,
            "n_biological_processes": len(self.prior),
            "total_unique_genes_in_prior": self.total_unique_genes,
            "avg_genes_per_process": np.mean(self.process_sizes),
            "min_genes_per_process": min(self.process_sizes),
            "max_genes_per_process": max(self.process_sizes),
            "processes_sampled_per_sample": self.n_processes,
            "target_context_window": self.context_window
        }
    
    def __del__(self):
        """
        Cleanup method to properly close the AnnData file.
        """
        if hasattr(self, 'adata') and self.adata is not None:
            try:
                # Close the backed AnnData file
                if hasattr(self.adata, 'file') and self.adata.file is not None:
                    self.adata.file.close()
            except:
                pass  # Ignore cleanup errors