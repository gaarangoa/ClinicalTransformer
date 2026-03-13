# Troubleshooting

## Training Issues

### CUDA out of memory

**Symptoms:** `RuntimeError: CUDA out of memory` during training.

**Solutions:**
1. Reduce `dataset.batch_size` in `config.yaml`
2. Decrease `dataset.context_window`
3. Enable gradient accumulation: set `trainer.accumulate_grad_batches` to 2, 4, or 8
4. Enable optimizer offloading: set `trainer.strategy.params.offload_optimizer: True`
5. Switch to DeepSpeed stage 3: set `trainer.strategy.params.stage: 3`

### NaN loss

**Symptoms:** Training loss becomes `NaN` after some epochs.

**Solutions:**
1. Switch precision from `16-mixed` to `bf16-mixed` (if your GPU supports bfloat16)
2. Increase `trainer.accumulate_grad_batches` to stabilise gradients
3. Lower `model.learning_rate`

### Training loss not decreasing

**Possible causes:**
- Learning rate too low or too high. Try values between `1e-5` and `1e-3`.
- Batch size too small. Increase `dataset.batch_size` or `trainer.accumulate_grad_batches`.
- Data encoding issue. Go back to {doc}`../vnbert/build-dataset` and verify the encoding looks correct.

### Slow data loading

**Solutions:**
1. Copy the `.h5ad` file to `/dev/shm/` (shared memory) for faster I/O
2. Increase `dataset.num_workers` (up to `num_cpus - 1`)
3. Make sure you're not running other I/O-heavy processes on the same disk

## Configuration Issues

### vocab_size mismatch

**Symptom:** Index error or embedding lookup failure during training.

**Fix:** Ensure `model.vocab_size` in `config.yaml` is greater than or equal to `tokenizer.vocab_size`. Check with:

```python
from clinical_transformer import vnBertTokenizerTabular
tokenizer = vnBertTokenizerTabular.from_pretrained("path/to/tokenizer")
print(tokenizer.vocab_size)
```

### devices mismatch

**Symptom:** Training hangs or crashes at startup.

**Fix:** `trainer.devices` in `config.yaml` must match `--nproc_per_node` in the `torchrun` command.

## Model Release Issues

### Can't load checkpoint

**Symptom:** Error when calling `LightningTrainerModel.load_from_checkpoint()`.

**Fix:** You must run the DeepSpeed conversion step first. See {doc}`../vnbert/release-model`.

### Missing tokenizer files

**Symptom:** `FileNotFoundError` when loading the tokenizer from the released model directory.

**Fix:** Make sure you copied the tokenizer into the model hub directory during the release step. The tokenizer is saved separately from the model weights.

## Inference Issues

### Feature name mismatch

**Symptom:** Tokenizer returns empty or incorrect encodings.

**Fix:** The feature names in your input dictionaries must exactly match the feature names used when fitting the tokenizer in {doc}`../vnbert/build-dataset`. Check with:

```python
print(tokenizer.vocab)  # or tokenizer.feature_encoder
```
