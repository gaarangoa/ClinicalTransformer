from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing import Dict, List, Optional, Union, Any
import numpy as np
import logging
import os
import json
from tqdm import tqdm

logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Tokenizer(PreTrainedTokenizer):
    """
    A Hugging Face compatible tokenizer for gene expression data.
    
    This tokenizer converts gene expression samples into token sequences
    and preserves the raw expression values.
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
        return_quantile_values: bool = False,
        return_zscore_values: bool = False,
        return_robust_zscore_values: bool = False,
        n_quantiles: int = 100,
        subset: Optional[Union[List[str], int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encode a single gene expression sample.
        
        Args:
            sample: Dictionary mapping gene names to expression values
            return_attention_mask: Whether to return attention mask
            return_quantile_values: Whether to return quantile normalized
                values
            return_zscore_values: Whether to return z-score normalized
                values (per-sample standardization)
            return_robust_zscore_values: Whether to return robust z-score
                normalized values using median and MAD
            n_quantiles: Number of quantiles for quantile normalization
            subset: Optional filter for tokens. Can be:
                   - List[str]: specific gene names to include
                   - int: top N genes to include (by expression value)
                   - None: include all valid genes (default)
            
        Returns:
            Dictionary with input_ids, raw_values, quantile_values (optional),
            zscore_values (optional), robust_zscore_values (optional),
            and attention_mask
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
                "raw_values": [],
                "quantile_values": [] if return_quantile_values else None,
                "zscore_values": [] if return_zscore_values else None,
                "robust_zscore_values": ([]
                                         if return_robust_zscore_values
                                         else None),
                "attention_mask": [] if return_attention_mask else None,
            }
        
        # Compute quantile normalization BEFORE subset filtering
        quantile_map = {}
        if return_quantile_values:
            # Create quantiles based on ALL valid values in the sample
            all_values = list(valid_features.values())
            sorted_values = sorted(all_values)
            n_values = len(sorted_values)
            
            # Create quantile mapping for all features
            for feature, value in valid_features.items():
                # Find the quantile for this value
                rank = sum(1 for v in sorted_values if v <= value)
                # Normalize to [0, 1] range
                quantile = min(rank / n_values, 1.0)
                # Scale to n_quantiles
                quantile_bin = int(quantile * (n_quantiles - 1))
                quantile_map[feature] = quantile_bin / (n_quantiles - 1)
        
        # Compute z-score normalization BEFORE subset filtering
        zscore_map = {}
        if return_zscore_values:
            # Calculate mean and std from ALL valid values in the sample
            all_values = list(valid_features.values())
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            
            # Handle case where std is zero (all values are the same)
            if std_val == 0:
                std_val = 1.0
            
            # Create z-score mapping for all features
            for feature, value in valid_features.items():
                zscore_map[feature] = (value - mean_val) / std_val
        
        # Compute robust z-score normalization BEFORE subset filtering
        robust_zscore_map = {}
        if return_robust_zscore_values:
            # Calculate median and MAD from ALL valid values in the sample
            all_values = np.array(list(valid_features.values()), dtype=np.float32)
            median_val = np.median(all_values)
            mad_val = np.median(np.abs(all_values - median_val)) + 1e-5
            
            # Vectorized robust z-score computation
            robust_zscores = ((all_values - median_val) / mad_val).astype(np.float32)
            
            # Create robust z-score mapping for all features
            for i, feature in enumerate(valid_features.keys()):
                robust_zscore_map[feature] = float(robust_zscores[i])
        
        # Apply subset filter if specified
        sample_items = list(valid_features.items())
        if subset is not None:
            if isinstance(subset, int):
                # Take top N genes by expression value (sorted descending)
                sample_items = sorted(sample_items,
                                      key=lambda x: x[1],
                                      reverse=True)[:subset]
            elif isinstance(subset, list):
                # Filter to only include specified gene names
                subset_set = set(subset)
                sample_items = [(feature, value)
                                for feature, value in sample_items
                                if feature in subset_set]
        
        input_ids = []
        raw_values = []
        quantile_values = []
        zscore_values = []
        robust_zscore_values = []
        
        for feature, value in sample_items:
            input_ids.append(self._vocab[feature])
            raw_values.append(value)
            if return_quantile_values:
                quantile_values.append(quantile_map[feature])
            if return_zscore_values:
                zscore_values.append(zscore_map[feature])
            if return_robust_zscore_values:
                robust_zscore_values.append(robust_zscore_map[feature])
        
        result = {
            "input_ids": input_ids,
            "raw_values": raw_values
        }
        
        if return_quantile_values:
            result["quantile_values"] = quantile_values
        
        if return_zscore_values:
            result["zscore_values"] = zscore_values
        
        if return_robust_zscore_values:
            result["robust_zscore_values"] = robust_zscore_values
        
        if return_attention_mask:
            result["attention_mask"] = [1] * len(input_ids)
        
        return result
    
    def __call__(
        self,
        samples: Union[Dict[str, float], List[Dict[str, float]]],
        return_tensors: Optional[str] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True,
        return_quantile_values: bool = False,
        return_zscore_values: bool = False,
        return_robust_zscore_values: bool = False,
        n_quantiles: int = 100,
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
            return_quantile_values: Whether to return quantile normalized
                values
            return_zscore_values: Whether to return z-score normalized
                values (per-sample standardization)
            return_robust_zscore_values: Whether to return robust z-score
                normalized values using median and MAD
            n_quantiles: Number of quantiles for quantile normalization
            subset: Optional filter for tokens. Can be:
                   - List[str]: specific gene names to include
                   - int: top N genes to include (by expression value)
                   - None: include all valid genes (default)
            
        Returns:
            BatchEncoding with input_ids, raw_values, quantile_values
            (optional), zscore_values (optional), robust_zscore_values
            (optional), and attention_mask
        """
        # Handle single sample vs batch
        is_batched = isinstance(samples, list)
        if not is_batched:
            samples = [samples]
        
        # Encode all samples
        batch_input_ids = []
        batch_attention_mask = []
        batch_raw_values = []
        batch_quantile_values = []
        batch_zscore_values = []
        batch_robust_zscore_values = []
        
        for sample in tqdm(samples, desc="Tokenizing samples"):
            encoded = self.encode_sample(
                sample,
                return_attention_mask=return_attention_mask,
                return_quantile_values=return_quantile_values,
                return_zscore_values=return_zscore_values,
                return_robust_zscore_values=return_robust_zscore_values,
                n_quantiles=n_quantiles,
                subset=subset
            )
            
            batch_input_ids.append(encoded["input_ids"])
            batch_raw_values.append(encoded["raw_values"])
            if return_attention_mask:
                batch_attention_mask.append(encoded["attention_mask"])
            if return_quantile_values:
                batch_quantile_values.append(encoded["quantile_values"])
            if return_zscore_values:
                batch_zscore_values.append(encoded["zscore_values"])
            if return_robust_zscore_values:
                batch_robust_zscore_values.append(
                    encoded["robust_zscore_values"])
        
        # Create result dictionary
        encoded_inputs = {
            "input_ids": batch_input_ids,
            "raw_values": batch_raw_values
        }
        
        if return_attention_mask:
            encoded_inputs["attention_mask"] = batch_attention_mask
        
        if return_quantile_values:
            encoded_inputs["quantile_values"] = batch_quantile_values
        
        if return_zscore_values:
            encoded_inputs["zscore_values"] = batch_zscore_values
        
        if return_robust_zscore_values:
            encoded_inputs["robust_zscore_values"] = batch_robust_zscore_values
        
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
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_mask = []
        padded_raw_values = []
        padded_quantile_values = []
        padded_zscore_values = []
        padded_robust_zscore_values = []
        
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
            
            # Pad raw_values with zeros
            if "raw_values" in encoded_inputs:
                raw_values = encoded_inputs["raw_values"][i]
                padded_raw = raw_values + [0.0] * padding_length
                padded_raw_values.append(padded_raw)
            
            # Pad quantile_values with zeros
            if "quantile_values" in encoded_inputs:
                quantile_values = encoded_inputs["quantile_values"][i]
                padded_quantile = quantile_values + [0.0] * padding_length
                padded_quantile_values.append(padded_quantile)
            
            # Pad zscore_values with zeros
            if "zscore_values" in encoded_inputs:
                zscore_values = encoded_inputs["zscore_values"][i]
                padded_zscore = zscore_values + [0.0] * padding_length
                padded_zscore_values.append(padded_zscore)
            
            # Pad robust_zscore_values with zeros
            if "robust_zscore_values" in encoded_inputs:
                robust_zscore_values = (
                    encoded_inputs["robust_zscore_values"][i])
                padded_robust_zscore = (robust_zscore_values +
                                        [0.0] * padding_length)
                padded_robust_zscore_values.append(padded_robust_zscore)
        
        # Update encoded_inputs
        encoded_inputs["input_ids"] = padded_input_ids
        if return_attention_mask and padded_attention_mask:
            encoded_inputs["attention_mask"] = padded_attention_mask
        if padded_raw_values:
            encoded_inputs["raw_values"] = padded_raw_values
        if padded_quantile_values:
            encoded_inputs["quantile_values"] = padded_quantile_values
        if padded_zscore_values:
            encoded_inputs["zscore_values"] = padded_zscore_values
        if padded_robust_zscore_values:
            encoded_inputs["robust_zscore_values"] = (
                padded_robust_zscore_values)
        
        return encoded_inputs
