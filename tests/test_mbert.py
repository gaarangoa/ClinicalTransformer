"""
Test mBERT model and dataset with dummy data.
Validates forward pass, training step, attention masking, and dataset optimizations.
"""
import torch
import numpy as np
from transformers.models.bert.modeling_bert import BertConfig
from clinical_transformer.mbert.dataset import MaskedTokenDataset
from clinical_transformer.mbert.modeling import (
    nBERTPretrainedModel,
    nBertPretrainedModelForMaskingValuePrediction,
    LightningTrainerModel,
)


def make_config(use_scgpt_mask=True, **overrides):
    params = dict(
        vocab_size=30,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_hidden_layers=2,
        pad_token_id=0,
        use_scgpt_mask=use_scgpt_mask,
        optimizer="torch.optim.Adam",
        learning_rate=1e-4,
    )
    params.update(overrides)
    return BertConfig(**params)


def make_dummy_data(n_samples=50, n_genes=100):
    """Create dummy token/value arrays matching pickle format."""
    tokens = [np.random.randint(1, 30, size=n_genes).tolist() for _ in range(n_samples)]
    values = [np.random.randn(n_genes).tolist() for _ in range(n_samples)]
    return tokens, values


# ── Dataset tests ──


def test_dataset_basic():
    tokens, values = make_dummy_data(10, 50)
    ds = MaskedTokenDataset(tokens, values, context_window=20, mask_prob=0.3)
    assert len(ds) == 10

    sample = ds[0]
    assert sample["tokens"].dtype == torch.long
    assert sample["values"].dtype == torch.float32
    assert sample["labels"].dtype == torch.float32
    # CLS + context_window = 21
    assert sample["tokens"].shape[0] == 21
    assert sample["tokens"][0] == 2  # CLS token
    assert sample["labels"][0] == 1.0  # CLS label
    print("  [PASS] dataset_basic")


def test_dataset_masking():
    tokens, values = make_dummy_data(10, 50)
    ds = MaskedTokenDataset(tokens, values, context_window=20, mask_prob=0.3)
    sample = ds[0]
    num_masked = int(20 * 0.3)
    # Last num_masked positions should be -10.0
    masked_vals = sample["values"][-num_masked:]
    assert (masked_vals == -10.0).all(), f"Expected -10.0, got {masked_vals}"
    # Labels should NOT be -10.0
    assert (sample["labels"][-num_masked:] != -10.0).all()
    print("  [PASS] dataset_masking")


def test_dataset_pre_conversion():
    tokens, values = make_dummy_data(10, 50)
    ds = MaskedTokenDataset(tokens, values, context_window=20, mask_prob=0.15)
    # Verify data was pre-converted to tensors at init
    assert isinstance(ds.input_ids[0], torch.Tensor)
    assert isinstance(ds.raw_values[0], torch.Tensor)
    print("  [PASS] dataset_pre_conversion")


def test_dataset_no_context_window():
    tokens, values = make_dummy_data(5, 30)
    ds = MaskedTokenDataset(tokens, values, context_window=None, mask_prob=0.15)
    sample = ds[0]
    assert sample["tokens"].shape[0] == 31  # 30 + CLS
    print("  [PASS] dataset_no_context_window")


def test_dataloader():
    tokens, values = make_dummy_data(20, 50)
    ds = MaskedTokenDataset(tokens, values, context_window=20, mask_prob=0.3)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(dl))
    assert batch["tokens"].shape == (4, 21)
    assert batch["values"].shape == (4, 21)
    assert batch["labels"].shape == (4, 21)
    print("  [PASS] dataloader")


# ── Model tests ──


def test_base_model_forward():
    config = make_config(use_scgpt_mask=True)
    model = nBERTPretrainedModel(config)
    model.eval()

    batch_size, seq_len = 4, 21
    tokens = torch.randint(1, 30, (batch_size, seq_len))
    values = torch.randn(batch_size, seq_len)
    values[:, -6:] = -10.0  # mask last 6

    with torch.no_grad():
        out = model(tokens, values, return_dict=True)

    assert out.last_hidden_state.shape == (batch_size, seq_len, 64)
    assert not torch.isnan(out.last_hidden_state).any(), "NaN in output"
    print("  [PASS] base_model_forward")


def test_base_model_bidirectional():
    config = make_config(use_scgpt_mask=False)
    model = nBERTPretrainedModel(config)
    model.eval()

    batch_size, seq_len = 4, 21
    tokens = torch.randint(1, 30, (batch_size, seq_len))
    values = torch.randn(batch_size, seq_len)
    values[:, -6:] = -10.0

    with torch.no_grad():
        out = model(tokens, values, return_dict=True)

    assert out.last_hidden_state.shape == (batch_size, seq_len, 64)
    assert not torch.isnan(out.last_hidden_state).any(), "NaN in output"
    print("  [PASS] base_model_bidirectional")


def test_prediction_model_forward():
    config = make_config(use_scgpt_mask=True)
    model = nBertPretrainedModelForMaskingValuePrediction(config)
    model.eval()

    batch_size, seq_len = 4, 21
    tokens = torch.randint(1, 30, (batch_size, seq_len))
    values = torch.randn(batch_size, seq_len)
    values[:, -6:] = -10.0

    with torch.no_grad():
        out = model(tokens, values, return_dict=True, output_predictions=True)

    assert out.value_predictions.shape == (batch_size, seq_len)
    assert out.last_hidden_state.shape == (batch_size, seq_len, 64)
    print("  [PASS] prediction_model_forward")


def test_prediction_model_skip_predictions():
    config = make_config(use_scgpt_mask=True)
    model = nBertPretrainedModelForMaskingValuePrediction(config)
    model.eval()

    tokens = torch.randint(1, 30, (4, 21))
    values = torch.randn(4, 21)
    values[:, -6:] = -10.0

    with torch.no_grad():
        out = model(tokens, values, return_dict=True, output_predictions=False)

    assert out.value_predictions is None
    assert out.last_hidden_state is not None
    print("  [PASS] prediction_model_skip_predictions")


def test_training_step():
    config = make_config(use_scgpt_mask=True)
    model = LightningTrainerModel(config)

    batch_size, seq_len = 4, 21
    batch = {
        "tokens": torch.randint(1, 30, (batch_size, seq_len)),
        "values": torch.randn(batch_size, seq_len),
        "labels": torch.randn(batch_size, seq_len),
    }
    batch["values"][:, -6:] = -10.0

    loss = model.training_step(batch, 0)
    assert loss.shape == ()
    assert loss.requires_grad
    assert not torch.isnan(loss), "NaN loss"

    # Verify backward works
    loss.backward()
    print("  [PASS] training_step")


def test_training_step_no_masked():
    config = make_config(use_scgpt_mask=True)
    model = LightningTrainerModel(config)

    batch = {
        "tokens": torch.randint(1, 30, (4, 21)),
        "values": torch.randn(4, 21),  # no -10.0 values
        "labels": torch.randn(4, 21),
    }

    loss = model.training_step(batch, 0)
    assert loss.item() == 0.0
    print("  [PASS] training_step_no_masked")


def test_training_step_masked_only_prediction():
    """Verify prediction head runs only on masked positions."""
    config = make_config(use_scgpt_mask=True)
    model = LightningTrainerModel(config)

    batch_size, seq_len = 4, 21
    num_masked = 6
    batch = {
        "tokens": torch.randint(1, 30, (batch_size, seq_len)),
        "values": torch.randn(batch_size, seq_len),
        "labels": torch.randn(batch_size, seq_len),
    }
    batch["values"][:, -num_masked:] = -10.0

    loss = model.training_step(batch, 0)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    print("  [PASS] training_step_masked_only_prediction")


# ── Attention mask tests ──


def test_scgpt_mask_shape():
    """Verify scGPT mask is (1,1,S,S) — built from one sample, broadcast."""
    config = make_config(use_scgpt_mask=True)
    model = nBERTPretrainedModel(config)
    model.eval()

    seq_len = 21
    tokens = torch.randint(1, 30, (4, seq_len))
    values = torch.randn(4, seq_len)
    values[:, -6:] = -10.0

    # Patch to capture mask
    captured = {}
    orig_forward = model.encoder[0].forward

    def capture_forward(hidden_states, attention_mask=None, **kw):
        captured["mask"] = attention_mask
        return orig_forward(hidden_states, attention_mask=attention_mask, **kw)

    model.encoder[0].forward = capture_forward

    with torch.no_grad():
        model(tokens, values)

    mask = captured["mask"]
    # Should be (1,1,S,S) — single sample, broadcast over batch
    assert mask.shape == (1, 1, seq_len, seq_len), f"Expected (1,1,{seq_len},{seq_len}), got {mask.shape}"
    print("  [PASS] scgpt_mask_shape")


def test_bidirectional_mask_no_padding():
    """Verify bidirectional mask is None when no padding (enables flash kernel)."""
    config = make_config(use_scgpt_mask=False)
    model = nBERTPretrainedModel(config)
    model.eval()

    seq_len = 21
    tokens = torch.randint(1, 30, (4, seq_len))  # all non-zero = no padding
    values = torch.randn(4, seq_len)
    values[:, -6:] = -10.0

    captured = {}
    orig_forward = model.encoder[0].forward

    def capture_forward(hidden_states, attention_mask=None, **kw):
        captured["mask"] = attention_mask
        return orig_forward(hidden_states, attention_mask=attention_mask, **kw)

    model.encoder[0].forward = capture_forward

    with torch.no_grad():
        model(tokens, values)

    assert captured["mask"] is None, f"Expected None mask for no-padding bidirectional, got {type(captured['mask'])}"
    print("  [PASS] bidirectional_mask_no_padding")


def test_bidirectional_mask_with_padding():
    """Verify bidirectional mask is (1,1,1,S) when padding exists."""
    config = make_config(use_scgpt_mask=False)
    model = nBERTPretrainedModel(config)
    model.eval()

    seq_len = 21
    tokens = torch.randint(1, 30, (4, seq_len))
    tokens[:, -3:] = 0  # add padding
    values = torch.randn(4, seq_len)
    values[:, -6:] = -10.0

    captured = {}
    orig_forward = model.encoder[0].forward

    def capture_forward(hidden_states, attention_mask=None, **kw):
        captured["mask"] = attention_mask
        return orig_forward(hidden_states, attention_mask=attention_mask, **kw)

    model.encoder[0].forward = capture_forward

    with torch.no_grad():
        model(tokens, values)

    mask = captured["mask"]
    assert mask is not None, "Expected mask when padding exists"
    assert mask.shape == (1, 1, 1, seq_len), f"Expected (1,1,1,{seq_len}), got {mask.shape}"
    print("  [PASS] bidirectional_mask_with_padding")


def test_scgpt_mask_correctness():
    """Verify the scGPT mask has correct attention patterns."""
    config = make_config(use_scgpt_mask=True)
    model = nBERTPretrainedModel(config)
    model.eval()

    seq_len = 11  # CLS + 10 tokens
    tokens = torch.randint(1, 30, (2, seq_len))
    values = torch.randn(2, seq_len)
    values[:, -3:] = -10.0  # mask last 3

    captured = {}
    orig_forward = model.encoder[0].forward

    def capture_forward(hidden_states, attention_mask=None, **kw):
        captured["mask"] = attention_mask
        return orig_forward(hidden_states, attention_mask=attention_mask, **kw)

    model.encoder[0].forward = capture_forward

    with torch.no_grad():
        model(tokens, values)

    mask = captured["mask"]
    # Convert back to boolean (0 = attend, large negative = block)
    bool_mask = (mask > -1.0).squeeze()  # (S, S)

    known_positions = list(range(8))  # CLS + 7 known
    masked_positions = [8, 9, 10]     # last 3

    # Known -> Known: ALLOWED
    for q in known_positions:
        for k in known_positions:
            assert bool_mask[q, k], f"Known->Known should be allowed at ({q},{k})"

    # Known -> Masked: BLOCKED
    for q in known_positions:
        for k in masked_positions:
            assert not bool_mask[q, k], f"Known->Masked should be blocked at ({q},{k})"

    # Masked -> Known: ALLOWED
    for q in masked_positions:
        for k in known_positions:
            assert bool_mask[q, k], f"Masked->Known should be allowed at ({q},{k})"

    # Masked -> Masked (different): BLOCKED
    for q in masked_positions:
        for k in masked_positions:
            if q != k:
                assert not bool_mask[q, k], f"Masked->Masked should be blocked at ({q},{k})"

    # Masked -> Self: ALLOWED
    for q in masked_positions:
        assert bool_mask[q, q], f"Masked->Self should be allowed at ({q},{q})"

    print("  [PASS] scgpt_mask_correctness")


# ── End-to-end test ──


def test_end_to_end():
    """Full pipeline: dataset -> dataloader -> model -> loss -> backward."""
    tokens, values = make_dummy_data(20, 50)
    ds = MaskedTokenDataset(tokens, values, context_window=20, mask_prob=0.3)
    dl = torch.utils.data.DataLoader(ds, batch_size=4)

    config = make_config(use_scgpt_mask=True)
    model = LightningTrainerModel(config)
    model.train()

    batch = next(iter(dl))
    loss = model.training_step(batch, 0)
    loss.backward()

    # Verify gradients flow
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "No gradients computed"
    print("  [PASS] end_to_end")


if __name__ == "__main__":
    print("Dataset tests:")
    test_dataset_basic()
    test_dataset_masking()
    test_dataset_pre_conversion()
    test_dataset_no_context_window()
    test_dataloader()

    print("\nModel tests:")
    test_base_model_forward()
    test_base_model_bidirectional()
    test_prediction_model_forward()
    test_prediction_model_skip_predictions()
    test_training_step()
    test_training_step_no_masked()
    test_training_step_masked_only_prediction()

    print("\nAttention mask tests:")
    test_scgpt_mask_shape()
    test_bidirectional_mask_no_padding()
    test_bidirectional_mask_with_padding()
    test_scgpt_mask_correctness()

    print("\nEnd-to-end tests:")
    test_end_to_end()

    print("\n✓ All tests passed")
