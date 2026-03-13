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


class TokenizerTabular(PreTrainedTokenizer):
    """
    A Hugging Face compatible tokenizer for mixed tabular data
    (numerical + categorical features).

    Workflow
    --------
    1.  ``fit(samples, categorical_features, numerical_features)``
        – learns vocabulary, categorical ordinal mappings, and per-feature
          statistics (min/max/mean/std/median/mad) from training data.

    2.  ``tokenizer(samples, ...)``  (i.e. ``__call__``)
        – encodes each sample into (input_ids, values) pairs where
          *input_ids* are feature token IDs and *values* are normalised
          scores, exactly matching the format expected by vnBERT-style
          models.

    Categorical handling
    --------------------
    Each categorical feature's levels are mapped to ordinal integers
    (0, 1, 2, …). These ordinals are then treated as pseudo-numerical
    values so they can be normalised (min-max or z-score) identically to
    genuinely numerical features.

    Numerical normalisation options (per-sample)
    ---------------------------------------------
    * **min-max** – ``(x - global_min) / (global_max - global_min)``
      using statistics learned at ``fit`` time.
    * **zscore** – ``(x - sample_mean) / sample_std`` computed within
      each sample.
    * **robust_zscore** – ``(x - sample_median) / (sample_MAD + ε)``
      computed within each sample.
    """

    vocab_files_names = {"vocab_file": "vocab.json"}
    CONFIG_NAME = "tabular_tokenizer_config.json"

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        cls_token: str = "<cls>",
        **kwargs,
    ):
        self.categorical_features: List[str] = categorical_features or []
        self.numerical_features: List[str] = numerical_features or []

        # Per-feature metadata populated during ``fit``
        self._feature_meta: Dict[str, Dict[str, Any]] = {}

        # Vocabulary
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
        else:
            self._vocab: Dict[str, int] = {}
            self._ids_to_tokens: Dict[int, str] = {}

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token=cls_token,
            **kwargs,
        )
        self._ensure_special_tokens_in_vocab()

    # ------------------------------------------------------------------
    # vocabulary helpers
    # ------------------------------------------------------------------
    def _ensure_special_tokens_in_vocab(self):
        if not self._vocab:
            self._vocab = {
                self.pad_token: 0,
                self.mask_token: 1,
                self.cls_token: 2,
            }
            if self.unk_token not in self._vocab:
                self._vocab[self.unk_token] = len(self._vocab)
            self._ids_to_tokens = {v: k for k, v in self._vocab.items()}

    def _load_vocab(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            self._vocab = json.load(f)
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> tuple:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        prefix = (filename_prefix + "-") if filename_prefix else ""
        vocab_file = os.path.join(
            save_directory, prefix + self.vocab_files_names["vocab_file"]
        )
        with open(vocab_file, "w") as f:
            json.dump(self._vocab, f, indent=2, sort_keys=True, ensure_ascii=False)
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab.copy()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(t) for t in tokens]

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(i) for i in ids]

    # ------------------------------------------------------------------
    # fit – learn vocab + feature statistics from training data
    # ------------------------------------------------------------------
    def fit(
        self,
        samples: List[Dict[str, Any]],
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
    ) -> "TokenizerTabular":
        """
        Learn vocabulary and per-feature statistics from training data.

        Parameters
        ----------
        samples : list of dict
            Each dict maps feature-name -> raw value.
        categorical_features : list of str, optional
            Which features to treat as categorical.
        numerical_features : list of str, optional
            Which features to treat as numerical.
        """
        if categorical_features is not None:
            self.categorical_features = list(categorical_features)
        if numerical_features is not None:
            self.numerical_features = list(numerical_features)

        all_features = self.categorical_features + self.numerical_features

        # --- 1. Build vocabulary: special tokens + feature names ---------
        self._vocab = {
            self.pad_token: 0,
            self.mask_token: 1,
            self.cls_token: 2,
        }
        for idx, feat in enumerate(all_features):
            self._vocab[feat] = idx + 3
        if self.unk_token not in self._vocab:
            self._vocab[self.unk_token] = len(self._vocab)
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}

        # --- 2. Initialise feature metadata ------------------------------
        self._feature_meta = {}
        for feat in self.categorical_features:
            self._feature_meta[feat] = {
                "type": "cat",
                "categories": {},       # cat_value -> ordinal int
                "categories_inv": {},   # ordinal int -> cat_value
            }
        for feat in self.numerical_features:
            self._feature_meta[feat] = {
                "type": "num",
                "min": float("inf"),
                "max": float("-inf"),
                "values": [],           # temporary – dropped after fit
            }

        # --- 3. First pass: collect values / categories ------------------
        for sample in tqdm(samples, desc="Fitting tokenizer (pass 1)"):
            for feat in all_features:
                if feat not in sample:
                    continue
                val = sample[feat]
                meta = self._feature_meta[feat]
                if meta["type"] == "num":
                    try:
                        fv = float(val)
                        if np.isnan(fv):
                            continue
                        meta["values"].append(fv)
                        if fv < meta["min"]:
                            meta["min"] = fv
                        if fv > meta["max"]:
                            meta["max"] = fv
                    except (ValueError, TypeError):
                        continue
                else:
                    # categorical – collect unique levels
                    val_str = str(val)
                    if val_str not in meta["categories"]:
                        meta["categories"][val_str] = len(meta["categories"])

        # --- 4. Finalise categorical mappings & compute stats -----------
        for feat in self.categorical_features:
            meta = self._feature_meta[feat]
            meta["categories_inv"] = {v: k for k, v in meta["categories"].items()}
            n_levels = len(meta["categories"])
            meta["min"] = 0.0
            meta["max"] = float(max(n_levels - 1, 1))
            # Compute mean/std/median/mad over ordinal codes
            ordinals = []
            for sample in samples:
                if feat in sample:
                    val_str = str(sample[feat])
                    if val_str in meta["categories"]:
                        ordinals.append(float(meta["categories"][val_str]))
            ordinals = np.array(ordinals, dtype=np.float32) if ordinals else np.array([0.0])
            meta["mean"] = float(np.mean(ordinals))
            meta["std"] = float(np.std(ordinals)) if len(ordinals) > 1 else 1.0
            meta["median"] = float(np.median(ordinals))
            meta["mad"] = float(np.median(np.abs(ordinals - meta["median"])))

        for feat in self.numerical_features:
            meta = self._feature_meta[feat]
            vals = np.array(meta["values"], dtype=np.float32) if meta["values"] else np.array([0.0])
            meta["mean"] = float(np.mean(vals))
            meta["std"] = float(np.std(vals)) if len(vals) > 1 else 1.0
            meta["median"] = float(np.median(vals))
            meta["mad"] = float(np.median(np.abs(vals - meta["median"])))
            del meta["values"]  # free memory

        logger.info(
            f"Fit complete – vocab_size={self.vocab_size}, "
            f"categorical={len(self.categorical_features)}, "
            f"numerical={len(self.numerical_features)}"
        )
        return self

    # ------------------------------------------------------------------
    # encode a single sample
    # ------------------------------------------------------------------
    def encode_sample(
        self,
        sample: Dict[str, Any],
        return_attention_mask: bool = True,
        return_minmax_values: bool = False,
        return_zscore_values: bool = False,
        return_robust_zscore_values: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Encode one sample (dict of feature->value) into token IDs + values.

        For categorical features the raw category string is first mapped to
        its ordinal integer.  Then all values (numerical and ordinalised
        categorical) are normalised with the requested method.

        Parameters
        ----------
        return_minmax_values : bool
            Global min-max normalisation using stats from ``fit``.
        return_zscore_values : bool
            Per-sample z-score: ``(x - mean) / std``.
        return_robust_zscore_values : bool
            Per-sample robust z-score: ``(x - median) / (MAD + eps)``.

        Returns
        -------
        dict with keys: input_ids, raw_values, and optionally
        minmax_values, zscore_values, robust_zscore_values, attention_mask.
        """
        all_features = self.categorical_features + self.numerical_features

        # --- step 1: resolve raw numeric value for every feature ---------
        resolved: Dict[str, float] = {}
        for feat in all_features:
            if feat not in sample:
                continue
            meta = self._feature_meta.get(feat)
            if meta is None:
                continue

            raw = sample[feat]

            if meta["type"] == "cat":
                val_str = str(raw)
                if val_str not in meta["categories"]:
                    continue
                resolved[feat] = float(meta["categories"][val_str])
            else:
                try:
                    fv = float(raw)
                    if np.isnan(fv):
                        continue
                    resolved[feat] = fv
                except (ValueError, TypeError):
                    continue

        if not resolved:
            return {
                "input_ids": [],
                "raw_values": [],
                "minmax_values": [] if return_minmax_values else None,
                "zscore_values": [] if return_zscore_values else None,
                "robust_zscore_values": [] if return_robust_zscore_values else None,
                "attention_mask": [] if return_attention_mask else None,
            }

        # --- step 2: compute normalisation maps --------------------------
        minmax_map: Dict[str, float] = {}
        if return_minmax_values:
            for feat, val in resolved.items():
                meta = self._feature_meta[feat]
                denom = meta["max"] - meta["min"]
                if denom == 0:
                    minmax_map[feat] = 1.0
                else:
                    minmax_map[feat] = (val - meta["min"]) / denom

        zscore_map: Dict[str, float] = {}
        if return_zscore_values:
            vals_arr = np.array(list(resolved.values()), dtype=np.float32)
            mu = float(np.mean(vals_arr))
            sigma = float(np.std(vals_arr))
            if sigma == 0:
                sigma = 1.0
            for feat, val in resolved.items():
                zscore_map[feat] = (val - mu) / sigma

        robust_zscore_map: Dict[str, float] = {}
        if return_robust_zscore_values:
            vals_arr = np.array(list(resolved.values()), dtype=np.float32)
            med = float(np.median(vals_arr))
            mad = float(np.median(np.abs(vals_arr - med))) + 1e-5
            for feat, val in resolved.items():
                robust_zscore_map[feat] = (val - med) / mad

        # --- step 3: build output arrays ---------------------------------
        input_ids: List[int] = []
        raw_values: List[float] = []
        minmax_values: List[float] = []
        zscore_values: List[float] = []
        robust_zscore_values: List[float] = []

        for feat, val in resolved.items():
            input_ids.append(self._vocab[feat])
            raw_values.append(val)
            if return_minmax_values:
                minmax_values.append(minmax_map[feat])
            if return_zscore_values:
                zscore_values.append(zscore_map[feat])
            if return_robust_zscore_values:
                robust_zscore_values.append(robust_zscore_map[feat])

        result: Dict[str, Any] = {
            "input_ids": input_ids,
            "raw_values": raw_values,
        }
        if return_minmax_values:
            result["minmax_values"] = minmax_values
        if return_zscore_values:
            result["zscore_values"] = zscore_values
        if return_robust_zscore_values:
            result["robust_zscore_values"] = robust_zscore_values
        if return_attention_mask:
            result["attention_mask"] = [1] * len(input_ids)
        return result

    # ------------------------------------------------------------------
    # __call__  –  batch encoding
    # ------------------------------------------------------------------
    def __call__(
        self,
        samples: Union[Dict[str, Any], List[Dict[str, Any]]],
        return_tensors: Optional[str] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True,
        return_minmax_values: bool = False,
        return_zscore_values: bool = False,
        return_robust_zscore_values: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenise one or more samples.

        Parameters mirror ``encode_sample`` plus standard HF arguments.
        """
        is_batched = isinstance(samples, list)
        if not is_batched:
            samples = [samples]

        batch_input_ids: List[List[int]] = []
        batch_raw_values: List[List[float]] = []
        batch_attention_mask: List[List[int]] = []
        batch_minmax: List[List[float]] = []
        batch_zscore: List[List[float]] = []
        batch_robust: List[List[float]] = []

        for sample in tqdm(samples, desc="Tokenizing samples"):
            enc = self.encode_sample(
                sample,
                return_attention_mask=return_attention_mask,
                return_minmax_values=return_minmax_values,
                return_zscore_values=return_zscore_values,
                return_robust_zscore_values=return_robust_zscore_values,
            )
            batch_input_ids.append(enc["input_ids"])
            batch_raw_values.append(enc["raw_values"])
            if return_attention_mask:
                batch_attention_mask.append(enc["attention_mask"])
            if return_minmax_values:
                batch_minmax.append(enc["minmax_values"])
            if return_zscore_values:
                batch_zscore.append(enc["zscore_values"])
            if return_robust_zscore_values:
                batch_robust.append(enc["robust_zscore_values"])

        encoded_inputs: Dict[str, Any] = {
            "input_ids": batch_input_ids,
            "raw_values": batch_raw_values,
        }
        if return_attention_mask:
            encoded_inputs["attention_mask"] = batch_attention_mask
        if return_minmax_values:
            encoded_inputs["minmax_values"] = batch_minmax
        if return_zscore_values:
            encoded_inputs["zscore_values"] = batch_zscore
        if return_robust_zscore_values:
            encoded_inputs["robust_zscore_values"] = batch_robust

        batch_encoding = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors
        )

        if padding and max_length:
            batch_encoding = self.pad(
                batch_encoding,
                padding=padding,
                max_length=max_length,
                return_attention_mask=return_attention_mask,
            )
        return batch_encoding

    # ------------------------------------------------------------------
    # padding
    # ------------------------------------------------------------------
    def pad(
        self,
        encoded_inputs: BatchEncoding,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        if max_length is None:
            max_length = max(len(s) for s in encoded_inputs["input_ids"])

        value_keys = [
            k for k in ("raw_values", "minmax_values", "zscore_values",
                        "robust_zscore_values")
            if k in encoded_inputs
        ]

        padded: Dict[str, list] = {k: [] for k in ["input_ids"] + value_keys}
        if return_attention_mask and "attention_mask" in encoded_inputs:
            padded["attention_mask"] = []

        for i, ids in enumerate(encoded_inputs["input_ids"]):
            pad_len = max_length - len(ids)
            padded["input_ids"].append(list(ids) + [self.pad_token_id] * pad_len)
            for vk in value_keys:
                padded[vk].append(list(encoded_inputs[vk][i]) + [0.0] * pad_len)
            if "attention_mask" in padded:
                padded["attention_mask"].append(
                    list(encoded_inputs["attention_mask"][i]) + [0] * pad_len
                )

        for k, v in padded.items():
            encoded_inputs[k] = v
        return encoded_inputs

    # ------------------------------------------------------------------
    # save / load (HF-compatible save_pretrained / from_pretrained)
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        # 1. vocab
        self.save_vocabulary(save_directory)

        # 2. config (feature metadata + feature lists)
        config = {
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "feature_meta": self._feature_meta,
        }
        config_path = os.path.join(save_directory, self.CONFIG_NAME)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Tokenizer saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs) -> "TokenizerTabular":
        vocab_file = os.path.join(pretrained_path, "vocab.json")
        config_file = os.path.join(pretrained_path, cls.CONFIG_NAME)

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config not found: {config_file}")

        with open(config_file, "r") as f:
            config = json.load(f)

        tok = cls(
            vocab_file=vocab_file,
            categorical_features=config["categorical_features"],
            numerical_features=config["numerical_features"],
            **kwargs,
        )
        tok._feature_meta = config["feature_meta"]
        return tok
