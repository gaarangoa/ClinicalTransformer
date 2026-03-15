"""
Test dBERT model: forward pass, disentangled attention mask, and training step.
"""
import torch
import numpy as np
from transformers.models.bert.modeling_bert import BertConfig
from clinical_transformer.dbert.dataset import MaskedTokenDataset, collate_variable_length
from clinical_transformer.dbert.modeling import (
    nBERTPretrainedModel,
    nBertPretrainedModelForMaskingValuePrediction,
    LightningTrainerModel,
)


def make_config(**overrides):
    params = dict(
        vocab_size=30,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_hidden_layers=2,
        pad_token_id=0,
        mask1_ratio=0.5,
        optimizer="torch.optim.Adam",
        learning_rate=1e-4,
    )
    params.update(overrides)
    return BertConfig(**params)


def make_dummy_data(n_samples=50, n_genes=100):
    tokens = [np.random.randint(1, 30, size=n_genes).tolist() for _ in range(n_samples)]
    values = [np.random.randn(n_genes).tolist() for _ in range(n_samples)]
    return tokens, values


def _capture_mask(model):
    """Patch first encoder layer to capture the attention mask."""
    captured = {}
    orig_forward = model.encoder[0].forward

    def capture_forward(hidden_states, attention_mask=None, **kw):
        captured["mask"] = attention_mask
        return orig_forward(hidden_states, attention_mask=attention_mask, **kw)

    model.encoder[0].forward = capture_forward
    return captured


# ── Dataset tests ──


def test_dataset_basic():
    tokens, values = make_dummy_data(10, 50)
    ds = MaskedTokenDataset(tokens, values, context_window=20, mask_prob=0.3)
    sample = ds[0]
    assert sample["tokens"].shape[0] == 21
    assert sample["tokens"][0] == 2
    print("  [PASS] dataset_basic")


# ── Model forward tests ──


def test_base_model_forward():
    config = make_config(mask1_ratio=0.5)
    model = nBERTPretrainedModel(config)
    model.eval()

    batch_size, seq_len = 4, 21
    tokens = torch.randint(1, 30, (batch_size, seq_len))
    values = torch.randn(batch_size, seq_len)
    values[:, -6:] = -10.0

    with torch.no_grad():
        out = model(tokens, values, return_dict=True)

    assert out.last_hidden_state.shape == (batch_size, seq_len, 64)
    assert not torch.isnan(out.last_hidden_state).any()
    print("  [PASS] base_model_forward")


def test_training_step():
    config = make_config(mask1_ratio=0.5)
    model = LightningTrainerModel(config)

    batch = {
        "tokens": torch.randint(1, 30, (4, 21)),
        "values": torch.randn(4, 21),
        "labels": torch.randn(4, 21),
    }
    batch["values"][:, -6:] = -10.0

    loss = model.training_step(batch, 0)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    loss.backward()
    print("  [PASS] training_step")


# ── Disentangled attention mask tests ──


def test_disentangled_mask_correctness():
    """Verify the 3-role attention pattern:
    Known   -> Known: ALLOWED,   -> Masked1: BLOCKED, -> Masked2: BLOCKED
    Masked1 -> Known: BLOCKED,   -> Masked1: BLOCKED, -> Self: ALLOWED
    Masked2 -> Known: ALLOWED,   -> Masked2: BLOCKED, -> Self: ALLOWED
    """
    config = make_config(mask1_ratio=0.5)
    model = nBERTPretrainedModel(config)
    model.eval()

    # CLS + 10 tokens, last 4 masked -> 2 mask1, 2 mask2
    seq_len = 11
    tokens = torch.randint(1, 30, (1, seq_len))
    values = torch.randn(1, seq_len)
    values[:, -4:] = -10.0  # positions 7,8,9,10 masked

    captured = _capture_mask(model)
    with torch.no_grad():
        model(tokens, values)

    mask = captured["mask"]
    bool_mask = (mask > -1.0).squeeze()  # (S, S)

    known = list(range(7))       # CLS + 6 known
    # mask1_ratio=0.5 -> first 2 of 4 masked = positions 7,8
    mask1 = [7, 8]
    mask2 = [9, 10]

    # Known -> Known: ALLOWED
    for q in known:
        for k in known:
            assert bool_mask[q, k], f"Known->Known should be allowed at ({q},{k})"

    # Known -> any masked: BLOCKED
    for q in known:
        for k in mask1 + mask2:
            assert not bool_mask[q, k], f"Known->Masked should be blocked at ({q},{k})"

    # Masked1 -> Known: BLOCKED (isolated)
    for q in mask1:
        for k in known:
            assert not bool_mask[q, k], f"Masked1->Known should be blocked at ({q},{k})"

    # Masked1 -> Self: ALLOWED
    for q in mask1:
        assert bool_mask[q, q], f"Masked1->Self should be allowed at ({q},{q})"

    # Masked1 -> other masked: BLOCKED
    for q in mask1:
        for k in mask1 + mask2:
            if q != k:
                assert not bool_mask[q, k], f"Masked1->other should be blocked at ({q},{k})"

    # Masked2 -> Known: ALLOWED (contextual)
    for q in mask2:
        for k in known:
            assert bool_mask[q, k], f"Masked2->Known should be allowed at ({q},{k})"

    # Masked2 -> Self: ALLOWED
    for q in mask2:
        assert bool_mask[q, q], f"Masked2->Self should be allowed at ({q},{q})"

    # Masked2 -> other masked: BLOCKED
    for q in mask2:
        for k in mask1 + mask2:
            if q != k:
                assert not bool_mask[q, k], f"Masked2->other should be blocked at ({q},{k})"

    print("  [PASS] disentangled_mask_correctness")


def test_mask1_ratio_zero_is_scgpt():
    """mask1_ratio=0.0 should produce scGPT-style mask (all masked see known)."""
    config = make_config(mask1_ratio=0.0)
    model = nBERTPretrainedModel(config)
    model.eval()

    seq_len = 11
    tokens = torch.randint(1, 30, (1, seq_len))
    values = torch.randn(1, seq_len)
    values[:, -4:] = -10.0

    captured = _capture_mask(model)
    with torch.no_grad():
        model(tokens, values)

    bool_mask = (captured["mask"] > -1.0).squeeze()
    known = list(range(7))
    masked = [7, 8, 9, 10]

    # All masked -> Known: ALLOWED (scGPT style)
    for q in masked:
        for k in known:
            assert bool_mask[q, k], f"mask1_ratio=0: Masked->Known should be allowed at ({q},{k})"

    print("  [PASS] mask1_ratio_zero_is_scgpt")


def test_mask1_ratio_one_is_fully_isolated():
    """mask1_ratio=1.0 should fully isolate all masked positions."""
    config = make_config(mask1_ratio=1.0)
    model = nBERTPretrainedModel(config)
    model.eval()

    seq_len = 11
    tokens = torch.randint(1, 30, (1, seq_len))
    values = torch.randn(1, seq_len)
    values[:, -4:] = -10.0

    captured = _capture_mask(model)
    with torch.no_grad():
        model(tokens, values)

    bool_mask = (captured["mask"] > -1.0).squeeze()
    masked = [7, 8, 9, 10]

    # All masked -> Known: BLOCKED
    for q in masked:
        for k in range(7):
            assert not bool_mask[q, k], f"mask1_ratio=1: Masked->Known should be blocked at ({q},{k})"
    # Self: ALLOWED
    for q in masked:
        assert bool_mask[q, q], f"mask1_ratio=1: Masked->Self should be allowed at ({q},{q})"

    print("  [PASS] mask1_ratio_one_is_fully_isolated")


# ── End-to-end test ──


def test_end_to_end():
    tokens, values = make_dummy_data(20, 50)
    ds = MaskedTokenDataset(tokens, values, context_window=20, mask_prob=0.3)
    dl = torch.utils.data.DataLoader(ds, batch_size=4)

    config = make_config(mask1_ratio=0.5)
    model = LightningTrainerModel(config)
    model.train()

    batch = next(iter(dl))
    loss = model.training_step(batch, 0)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "No gradients computed"
    print("  [PASS] end_to_end")


if __name__ == "__main__":
    print("Dataset tests:")
    test_dataset_basic()

    print("\nModel tests:")
    test_base_model_forward()
    test_training_step()

    print("\nDisentangled mask tests:")
    test_disentangled_mask_correctness()
    test_mask1_ratio_zero_is_scgpt()
    test_mask1_ratio_one_is_fully_isolated()

    print("\nEnd-to-end tests:")
    test_end_to_end()

    print("\nAll tests passed")
