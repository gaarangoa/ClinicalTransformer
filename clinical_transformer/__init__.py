"""
Clinical Transformer Library

Simplified imports for clinical transformer models and tokenizers.
"""

# ── vnBERT (primary API) ─────────────────────────────────────────────
from clinical_transformer.vnbert import (
    vnBertPretrainedModel,
    vnBertPretrainedModelForMVP,
    vnBertTokenizer,
    vnBertTokenizerTabular,
    MaskedTokenDataset as vnBertDatasetForMVP,
    LightningTrainerModel,
)

# ── Legacy models (backward-compatible aliases) ──────────────────────
from clinical_transformer._legacy.datasets.preprocessor.tabular import Preprocessor as nBertTokenizer
from clinical_transformer._legacy.training.BERT.nBERT import CTBERT as nBertPretrainedModel
from clinical_transformer._legacy.datasets.preprocessor.tabular_gpt import PreprocessorGPT as ModernBertRankTokenizer
from clinical_transformer._legacy.training.rnaBERT.modeling import nBERTPretrainedModel as rnBertPretrainedModel
from clinical_transformer._legacy.training.rnaBERT.tokenizer import GeneTokenizer as rnBertTokenizer
from clinical_transformer._legacy.training.ClinicalTransformerMIL.modeling import ClinicalTransformerMILConfig, ClinicalTransformerMILModel

# Export the main interfaces
__all__ = [
    # vnBERT components (primary)
    "vnBertPretrainedModel",
    "vnBertPretrainedModelForMVP",
    "vnBertTokenizer",
    "vnBertTokenizerTabular",
    "vnBertDatasetForMVP",
    "LightningTrainerModel",

    # Legacy model aliases
    "nBertTokenizer",
    "nBertPretrainedModel",
    "ModernBertRankTokenizer",
    "rnBertPretrainedModel",
    "rnBertTokenizer",
    "ClinicalTransformerMILConfig",
    "ClinicalTransformerMILModel",
]
