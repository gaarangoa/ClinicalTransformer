"""
Clinical Transformer Library

Simplified imports for clinical transformer models and tokenizers.
"""

# Import the main tokenizer and model classes for easy access
from clinical_transformer.pt.datasets.preprocessor.tabular import Preprocessor as nBertTokenizer
from clinical_transformer.pt.training.BERT.nBERT import CTBERT as nBertPretrainedModel
from clinical_transformer.pt.datasets.preprocessor.tabular_gpt import PreprocessorGPT as ModernBertRankTokenizer
from clinical_transformer.pt.training.rnaBERT.modeling import nBERTPretrainedModel as rnBertPretrainedModel
from clinical_transformer.pt.training.rnaBERT.tokenizer import GeneTokenizer as rnBertTokenizer
from clinical_transformer.pt.training.vnBERT.tokenizer import Tokenizer as vnBertTokenizer
from clinical_transformer.pt.training.vnBERT.modeling import nBertPretrainedModelForMaskingValuePrediction as vnBertPretrainedModelForMVP
from clinical_transformer.pt.training.vnBERT.dataset import MaskedTokenDataset as vnBertDatasetForMVP
from clinical_transformer.pt.training.vnBERT.modeling import nBERTPretrainedModel as vnBertPretrainedModel
from clinical_transformer.pt.training.ClinicalTransformerMIL.modeling import ClinicalTransformerMILConfig, ClinicalTransformerMILModel
from clinical_transformer.pt.training.vnBERT.tokenizer_tabular import TokenizerTabular as vnBertTokenizerTabular

# Export the main interfaces
__all__ = [
    # nBERT components
    "nBertTokenizer",
    "nBertPretrainedModel",

    # ModernBERT components
    "ModernBertRankTokenizer",
    
    # rnBERT components
    "rnBertPretrainedModel",
    "rnBertTokenizer",
    
    # VN-BERT components
    "vnBertTokenizer",
    "vnBertPretrainedModelForMVP",
    "vnBertDatasetForMVP",
    "vnBertPretrainedModel",
    "vnBertTokenizerTabular"

    # MIL model
    "ClinicalTransformerMILConfig",
    "ClinicalTransformerMILModel"
]
