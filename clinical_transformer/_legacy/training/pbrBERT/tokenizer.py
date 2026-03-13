import csv
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional, Union
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class GeneExpressionTokenizer(PreTrainedTokenizer):
    """
    Tokenizer for gene expression data that sorts genes by expression levels
    and returns normalized positions along with token IDs.
    
    This tokenizer filters out zero and empty expression values, sorts genes by 
    expression levels (highest to lowest), and returns both token IDs and 
    normalized position values for use in transformer models.
    
    Example:
        ```python
        # Initialize tokenizer with gene vocabulary
        gene_list = ["GENE1", "GENE2", "GENE3", "GENE4"]
        tokenizer = GeneExpressionTokenizer(gene_vocabulary=gene_list)
        
        # Tokenize a single sample
        sample_data = {"GENE1": 5.2, "GENE2": 0.0, "GENE3": 8.1, "GENE4": 2.3}
        token_ids, positions = tokenizer.tokenize_sample(sample_data)
        # Returns: ([gene_id_for_GENE3, gene_id_for_GENE1, gene_id_for_GENE4], [1.0, 0.67, 0.33])
        # Order: GENE3(8.1), GENE1(5.2), GENE4(2.3) - highest to lowest expression
        
        # Process multiple samples
        samples = [
            {"GENE1": 5.2, "GENE2": 0.0, "GENE3": 8.1, "GENE4": 2.3},
            {"GENE1": 1.1, "GENE2": 3.4, "GENE3": 0.0, "GENE4": 7.8}
        ]
        batch_encoding = tokenizer(samples, return_tensors="pt")
        
        # Process CSV file
        results = tokenizer.process_csv_file("expression_data.csv")
        ```
    """
    
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    def __init__(
        self, 
        gene_vocabulary: Optional[List[str]] = None,
        vocab_file: Optional[str] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>", 
        mask_token: str = "<mask>",
        cls_token: str = "<cls>",
        **kwargs
    ):
        """
        Initialize tokenizer with gene vocabulary.
        
        Args:
            gene_vocabulary: List of gene names/IDs to include in vocabulary
            vocab_file: Path to vocabulary file
        """
        # Initialize vocabulary
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
        elif gene_vocabulary:
            self._build_vocab(gene_vocabulary)
        else:
            self._vocab = {}
            self._ids_to_tokens = {}
        
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token=cls_token,
            **kwargs
        )
        
        self.gene_vocabulary = set(gene_vocabulary) if gene_vocabulary else set()
    
    def _build_vocab(self, gene_vocabulary: List[str]):
        """Build vocabulary from gene list."""
        self._vocab = {
            "<pad>": 0,
            "<mask>": 1, 
            "<cls>": 2,
            "<unk>": 3
        }
        
        for i, gene in enumerate(gene_vocabulary, start=4):
            self._vocab[gene] = i
        
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
    
    def _load_vocab(self, vocab_file: str):
        """Load vocabulary from file."""
        with open(vocab_file, 'r') as f:
            self._vocab = json.load(f)
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """Save vocabulary to file."""
        if not os.path.isdir(save_directory):
            return
        
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        
        with open(vocab_file, 'w') as f:
            json.dump(self._vocab, f, indent=2)
        
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
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text (required by base class)."""
        return [text] if text in self._vocab else [self.unk_token]
    
    def tokenize_sample(self, gene_expressions: Dict[str, float]) -> Tuple[List[int], List[float]]:
        """
        Tokenize a single sample by sorting genes by expression and returning normalized positions.
        
        Args:
            gene_expressions: Dictionary mapping gene names to expression values
            
        Returns:
            Tuple of (token_ids, normalized_positions)
        """
        # Filter genes that are in vocabulary and have non-zero expression
        valid_genes = {gene: expr for gene, expr in gene_expressions.items() 
                      if gene in self.gene_vocabulary and expr != 0.0}
        
        if not valid_genes:
            return [self.convert_tokens_to_ids(self.pad_token)], [0.0]
        
        # Sort genes by expression (highest to lowest)
        sorted_genes = sorted(valid_genes.items(), key=lambda x: x[1], reverse=True)
        
        # Get token IDs and calculate normalized positions
        token_ids = [self.convert_tokens_to_ids(gene) for gene, _ in sorted_genes]
        num_genes = len(sorted_genes)
        normalized_positions = [(num_genes - i) / num_genes for i in range(num_genes)]
        
        return token_ids, normalized_positions
    
    def process_csv_file(
        self, 
        csv_file_path: str, 
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        return_gene_values: bool = True
    ) -> BatchEncoding:
        """
        Process CSV file line by line and tokenize each sample.
        
        Args:
            csv_file_path: Path to CSV file with samples as rows and genes as columns
            return_tensors: Type of tensors to return ('pt' for PyTorch, 'tf' for TensorFlow)
            return_attention_mask: Whether to return attention masks
            return_gene_values: Whether to return normalized positions
            
        Returns:
            BatchEncoding with input_ids, attention_mask, and gene_values
        """
        samples = []
        
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            
            # Read header (gene names)
            header = next(reader)
            gene_names = header
            
            # Process each sample
            for row in reader:
                if len(row) != len(gene_names):
                    continue  # Skip malformed rows
                
                # Create gene expression dictionary
                gene_expressions = {}
                for gene_name, expression_str in zip(gene_names, row):
                    try:
                        if expression_str.strip():  # Check if not empty
                            expression_value = float(expression_str)
                            if expression_value != 0.0:  # Check if not zero
                                gene_expressions[gene_name] = expression_value
                    except ValueError:
                        continue  # Skip invalid expression values
                
                samples.append(gene_expressions)
        
        # Use the __call__ method to process all samples
        return self(
            samples,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            return_gene_values=return_gene_values
        )

    # ...existing code...
    def __call__(
        self,
        samples: Union[Dict[str, float], List[Dict[str, float]]],
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        return_gene_values: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Main tokenization method.
        
        Args:
            samples: Single sample or list of samples
            return_tensors: Type of tensors to return
            return_attention_mask: Whether to return attention masks
            return_gene_values: Whether to return normalized positions
            
        Returns:
            BatchEncoding with input_ids, attention_mask, and gene_values
        """
        is_batched = isinstance(samples, list)
        if not is_batched:
            samples = [samples]
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_gene_values = []
        
        for sample in samples:
            token_ids, positions = self.tokenize_sample(sample)
            
            batch_input_ids.append(token_ids)
            if return_attention_mask:
                batch_attention_mask.append([1] * len(token_ids))
            if return_gene_values:
                batch_gene_values.append(positions)
        
        encoded_inputs = {"input_ids": batch_input_ids}
        if return_attention_mask:
            encoded_inputs["attention_mask"] = batch_attention_mask
        if return_gene_values:
            encoded_inputs["gene_values"] = batch_gene_values
        
        return BatchEncoding(encoded_inputs, tensor_type=return_tensors)
    
    def get_special_token_id(self, token: str) -> Optional[int]:
        """Get the ID of a special token."""
        special_tokens = {
            "<pad>": 0,
            "<mask>": 1,
            "<cls>": 2
        }
        return special_tokens.get(token)
