# vnBERT — Value-based BERT for Gene Expression and Tabular Data

## Attention Masking Strategy

vnBERT uses a modified attention masking strategy that differs from the original scGPT design. During masked value prediction, masked tokens are **fully isolated** in the attention mechanism — they can only attend to themselves and cannot gather context from visible (known) tokens.

The resulting attention pattern:

|              | → Known | → Masked | → Self |
|--------------|---------|----------|--------|
| **Known**    | ✅      | ❌       | ✅     |
| **Masked**   | ❌      | ❌       | ✅     |

This means:
- Known tokens have full bidirectional attention among themselves.
- Masked tokens receive no contextual information from other positions.
- No information leaks from masked positions to known positions.

### Design Rationale

This isolation creates two decoupled learning pathways within the same model:

**1. Visible pathway (attention-based)**
Known tokens attend to each other through the full transformer encoder. The CLS token aggregates a rich sample-level representation. Visible token hidden states capture conditional gene-gene relationships through learned attention weights.

**2. Masked pathway (identity-based)**
Masked tokens must predict their expression value using only their token embedding (gene identity) and the learned "unknown value" representation (`value_embeddings(0.0)`). The prediction passes through N transformer layers (MLP + LayerNorm + residual connections) but without any attention context.

### Why This Works

**Forced embedding quality.** Because masked tokens receive no contextual signal, the token embedding table must encode everything needed to predict a gene's expression value from identity alone. Genes with similar expression patterns across samples receive similar gradient updates, pushing their embeddings closer. This produces an embedding geometry where co-expressed genes cluster naturally — effectively contrastive learning on gene behavior without an explicit contrastive loss.

**Regularization by simplification.** Models that allow masked tokens to see context during pretraining can learn shallow co-expression shortcuts ("gene A is high, so gene B is probably high") instead of learning robust gene identity representations. Isolating masked tokens forces maximally informative embeddings that don't depend on which other genes happen to be visible.

**Stable optimization on the prediction path.** In standard masked prediction, the masked token's representation is a weighted sum over visible tokens — if attention weights are noisy (common early in training), prediction quality suffers. The isolated design bypasses this: the prediction path is a clean `embedding → MLP stack → value` pipeline that is straightforward to optimize.

**Full attention at inference time.** During inference, no tokens are masked — all positions are visible and benefit from the well-trained bidirectional attention mechanism. The isolation only affects pretraining. Downstream tasks that use CLS or mean-pooled embeddings get the full benefit of both the strong token embeddings and the learned attention patterns.

**Implicit denoising autoencoder.** The model acts as a denoising autoencoder where the "corruption" is complete information removal rather than partial noise. This is a stronger form of regularization than partial masking with context, forcing gene embeddings to be robust to total context dropout.

### Relationship to scGPT

The original scGPT attention strategy allows masked tokens to attend to known tokens (gathering context for prediction) while still preventing known tokens from seeing masked positions and masked tokens from seeing each other. vnBERT's stricter isolation is a deliberate departure from this design, trading contextual prediction for stronger embedding priors. See `mbert` for an implementation with the standard scGPT masking strategy.
