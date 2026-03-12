from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing import Dict, List, Optional, Union, Any
import numpy as np
import logging
import yaml
import os
import json
from tqdm import tqdm

logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class GeneTokenizer(PreTrainedTokenizer):
    """
    A Hugging Face compatible tokenizer for gene expression data.
    
    This tokenizer performs rank-based normalization where each sample's gene values 
    are sorted, ranked, and then normalized by dividing by the maximum rank in that sample.
    
    This creates a rank-normalized representation where values represent the relative ranking
    of genes within each sample, scaled between 0 and 1.
    """
    
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        numerical_features: Optional[List[str]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        cls_token: str = "<cls>",
        **kwargs
    ):
        self.numerical_features = numerical_features or []
        
        # Initialize vocabulary first
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
        else:
            self._vocab = {}
            self._ids_to_tokens = {}
        
        # Call parent constructor which will set up special tokens
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token=cls_token,
            **kwargs
        )
        
        # Ensure special tokens are in vocab if not already present
        self._ensure_special_tokens_in_vocab()
    
    def _ensure_special_tokens_in_vocab(self):
        """Ensure special tokens are properly added to vocabulary."""
        special_tokens_map = {
            self.pad_token: 0,
            self.mask_token: 1,
            self.cls_token: 2,
        }
        
        # Add special tokens if vocab is empty or doesn't contain them
        if not self._vocab:
            self._vocab = special_tokens_map.copy()
            if self.unk_token not in self._vocab:
                self._vocab[self.unk_token] = len(self._vocab)
            self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
    
    def _load_vocab(self, vocab_file: str):
        """Load vocabulary from file."""
        with open(vocab_file, 'r') as f:
            self._vocab = json.load(f)
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """Save vocabulary to file."""
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        
        with open(vocab_file, 'w') as f:
            json.dump(self._vocab, f, indent=2, sort_keys=True, ensure_ascii=False)
        
        return (vocab_file,)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary dictionary."""
        return self._vocab.copy()
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to id."""
        return self._vocab.get(token, self._vocab.get(self.unk_token, 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert id to token."""
        return self._ids_to_tokens.get(index, self.unk_token)
    
    def build_vocab(self, gene_features: List[str]):
        """
        Build vocabulary from gene features.
        
        Args:
            gene_features: List of gene feature names
        """
        logger.info('Building vocabulary from gene features...')
        
        # Create vocabulary with special tokens first
        self._vocab = {
            self.pad_token: 0,
            self.mask_token: 1,
            self.cls_token: 2,
        }
        
        # Add gene features starting from index 3
        for idx, gene in enumerate(gene_features):
            self._vocab[gene] = idx + 3
        
        # Add unk token if not present
        if self.unk_token not in self._vocab:
            self._vocab[self.unk_token] = len(self._vocab)
        
        # Create reverse mapping
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
        
        logger.info(f'Vocabulary built with {len(self._vocab)} tokens')
    
    def fit(self, gene_features: List[str]):
        """
        Fit the tokenizer on gene features.
        
        Args:
            gene_features: List of gene feature names to build vocabulary from
        """
        self.numerical_features = gene_features
        self.build_vocab(gene_features)
        return self
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert token(s) to id(s) using the parent class method."""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """Convert id(s) to token(s) using the parent class method."""
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(id_) for id_ in ids]
    
    def encode_sample(
        self, 
        sample: Dict[str, float], 
        return_attention_mask: bool = True,
        return_gene_values: bool = True,
        subset: Optional[Union[List[str], int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encode a single gene expression sample with rank-based normalization.
        
        Args:
            sample: Dictionary mapping gene names to expression values
            return_attention_mask: Whether to return attention mask
            return_gene_values: Whether to return normalized gene values
            subset: Optional filter for tokens. Can be:
                   - List[str]: specific gene names to include
                   - int: top N genes to include (by expression value)
                   - None: include all valid genes (default)
            
        Returns:
            Dictionary with input_ids, attention_mask (optional), and
            gene_values (optional)
        """
        # Filter out zero and NaN values
        valid_features = {}
        for feature, value in sample.items():
            try:
                float_value = float(value)
                # Skip if NaN, zero, or empty
                if np.isnan(float_value) or float_value == 0.0:
                    continue
            except (ValueError, TypeError):
                continue
            
            # Check if feature exists in vocabulary
            if feature not in self._vocab:
                continue
            
            valid_features[feature] = float_value
        
        if not valid_features:
            # Return empty encoding if no valid features
            return {
                "input_ids": [],
                "attention_mask": [] if return_attention_mask else None,
                "gene_values": [] if return_gene_values else None,
            }
        
        # Second pass: sort by values to get ranks
        # Sort features by their values (ascending order for ranking)
        sorted_features = sorted(valid_features.items(), key=lambda x: x[1])
        
        # Create rank mapping
        feature_ranks = {}
        for rank, (feature, value) in enumerate(sorted_features):
            feature_ranks[feature] = rank + 1  # ranks start from 1
        
        max_rank = len(sorted_features)
        
        # Third pass: create tokens and normalized rank values
        # Sort by original values (descending) to maintain ordering
        sample_items = sorted(valid_features.items(),
                             key=lambda x: x[1], reverse=True)
        
        # Apply subset filter if specified
        if subset is not None:
            if isinstance(subset, int):
                # Take top N genes by expression value (sorted descending)
                sample_items = sample_items[:subset]
            elif isinstance(subset, list):
                # Filter to only include specified gene names
                subset_set = set(subset)
                sample_items = [(feature, value)
                               for feature, value in sample_items
                               if feature in subset_set]
        
        input_ids = []
        gene_values = []
        
        for feature, original_value in sample_items:
            input_ids.append(self._vocab[feature])
            # Normalize rank by max rank to get value between 0 and 1
            normalized_rank = feature_ranks[feature] / max_rank
            gene_values.append(normalized_rank)
        
        result = {"input_ids": input_ids}
        
        if return_attention_mask:
            result["attention_mask"] = [1] * len(input_ids)
        
        if return_gene_values:
            result["gene_values"] = gene_values
        
        return result
    
    def __call__(
        self,
        samples: Union[Dict[str, float], List[Dict[str, float]]],
        return_tensors: Optional[str] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True,
        return_gene_values: bool = True,
        subset: Optional[Union[List[str], int]] = None,
        **kwargs
    ) -> BatchEncoding:
        """
        Main tokenization method that handles both single samples and batches.
        
        Args:
            samples: Single sample or list of samples (gene expression dicts)
            return_tensors: Type of tensors to return ('pt'/'tf')
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_attention_mask: Whether to return attention masks
            return_gene_values: Whether to return normalized gene values
            subset: Optional filter for tokens. Can be:
                   - List[str]: specific gene names to include
                   - int: top N genes to include (by expression value)
                   - None: include all valid genes (default)
            
        Returns:
            BatchEncoding with input_ids, attention_mask, and gene_values
        """
        # Handle single sample vs batch
        is_batched = isinstance(samples, list)
        if not is_batched:
            samples = [samples]
        
        # Encode all samples
        batch_input_ids = []
        batch_attention_mask = []
        batch_gene_values = []
        
        for sample in samples:
            encoded = self.encode_sample(
                sample,
                return_attention_mask=return_attention_mask,
                return_gene_values=return_gene_values,
                subset=subset
            )
            
            batch_input_ids.append(encoded["input_ids"])
            if return_attention_mask:
                batch_attention_mask.append(encoded["attention_mask"])
            if return_gene_values:
                batch_gene_values.append(encoded["gene_values"])
        
        # Create result dictionary
        encoded_inputs = {"input_ids": batch_input_ids}
        
        if return_attention_mask:
            encoded_inputs["attention_mask"] = batch_attention_mask
        
        if return_gene_values:
            encoded_inputs["gene_values"] = batch_gene_values
        
        # Convert to BatchEncoding
        batch_encoding = BatchEncoding(
            encoded_inputs,
            tensor_type=return_tensors,
        )
        
        # Handle padding if requested
        if padding and max_length:
            batch_encoding = self.pad(
                batch_encoding,
                padding=padding,
                max_length=max_length,
                return_attention_mask=return_attention_mask,
            )
        
        return batch_encoding
    
    def pad(
        self,
        encoded_inputs: BatchEncoding,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Pad sequences to the same length.
        """
        if max_length is None:
            max_length = max(len(seq) for seq in encoded_inputs["input_ids"])
        
        # Pad input_ids
        padded_input_ids = []
        padded_attention_mask = []
        padded_gene_values = []
        
        for i, input_ids in enumerate(encoded_inputs["input_ids"]):
            # Calculate padding needed
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids with pad_token_id
            padded_ids = input_ids + [self.pad_token_id] * padding_length
            padded_input_ids.append(padded_ids)
            
            # Pad attention_mask
            if return_attention_mask and "attention_mask" in encoded_inputs:
                attention_mask = encoded_inputs["attention_mask"][i]
                padded_mask = attention_mask + [0] * padding_length
                padded_attention_mask.append(padded_mask)
            
            # Pad gene_values with zeros
            if "gene_values" in encoded_inputs:
                gene_values = encoded_inputs["gene_values"][i]
                padded_values = gene_values + [0.0] * padding_length
                padded_gene_values.append(padded_values)
        
        # Update encoded_inputs
        encoded_inputs["input_ids"] = padded_input_ids
        if return_attention_mask and padded_attention_mask:
            encoded_inputs["attention_mask"] = padded_attention_mask
        if padded_gene_values:
            encoded_inputs["gene_values"] = padded_gene_values
        
        return encoded_inputs
