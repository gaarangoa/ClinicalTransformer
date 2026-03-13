# pbrBERT: Gene Expression Tokenizer and Dataset

This directory contains components for processing gene expression data for transformer-based models, specifically designed for clinical and genomic applications.

## Overview

The pbrBERT (prior burl rna Encoder Representations from Transformers) implementation includes:

- **`GeneExpressionTokenizer`**: Converts gene expression data into tokenized sequences
- **`MaskedTokenDataset`**: Creates masked language modeling datasets for self-supervised learning
- **Complete pipeline**: From raw gene expression values to training-ready batches

## Features

### GeneExpressionTokenizer
- ✅ Sorts genes by expression level (highest to lowest)
- ✅ Filters out zero/missing expression values
- ✅ Returns normalized position values alongside token IDs
- ✅ Supports batch processing
- ✅ Compatible with HuggingFace tokenizer interface

### MaskedTokenDataset
- ✅ Context window sampling for variable-length sequences
- ✅ Masked language modeling with configurable masking fraction
- ✅ Optional CLS token addition with correct sequence length handling
- ✅ Padding for sequences shorter than context window
- ✅ PyTorch DataLoader compatibility

## Quick Start

### Installation Requirements

```bash
pip install torch transformers numpy
```

### Basic Usage Example

```python
import torch
from torch.utils.data import DataLoader

# Import the components
from clinical_transformer.pt.training.pbrBERT.tokenizer import GeneExpressionTokenizer
from clinical_transformer.pt.training.pbrBERT.dataset import MaskedTokenDataset

# Step 1: Create gene expression data
gene_names = ["BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "PIK3CA", "PTEN"]

samples = [
    {"BRCA1": 8.5, "BRCA2": 7.2, "TP53": 9.1, "EGFR": 2.3, "KRAS": 0.0, "PIK3CA": 3.4, "PTEN": 6.7},
    {"BRCA1": 2.1, "BRCA2": 3.4, "TP53": 1.8, "EGFR": 9.2, "KRAS": 8.7, "PIK3CA": 7.9, "PTEN": 2.3},
    {"BRCA1": 5.2, "BRCA2": 0.0, "TP53": 6.8, "EGFR": 4.1, "KRAS": 0.0, "PIK3CA": 5.7, "PTEN": 4.9},
]

# Step 2: Initialize tokenizer
tokenizer = GeneExpressionTokenizer(gene_vocabulary=gene_names)

# Step 3: Tokenize data
batch_encoding = tokenizer(samples, return_tensors=None)
tokens_list = batch_encoding['input_ids']
values_list = batch_encoding['gene_values']

# Step 4: Create dataset
dataset = MaskedTokenDataset(
    tokens=tokens_list,
    values=values_list,
    context_window=5,
    return_cls=True,
    masking_fraction=0.15
)

# Step 5: Create DataLoader
def collate_fn(batch):
    tokens = torch.stack([item['tokens'] for item in batch])
    values = torch.stack([item['values'] for item in batch]) 
    original_tokens = torch.stack([item['original_tokens'] for item in batch])
    return {'tokens': tokens, 'values': values, 'original_tokens': original_tokens}

dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Step 6: Use in training loop
for batch in dataloader:
    tokens = batch['tokens']          # Shape: [batch_size, seq_len]
    values = batch['values']          # Shape: [batch_size, seq_len] 
    original_tokens = batch['original_tokens']  # For loss calculation
    
    print(f"Batch tokens shape: {tokens.shape}")
    print(f"Sample tokens: {tokens[0].tolist()}")
    break
```


## Complete Usage Example: nBERT Model Integration

Here's a comprehensive example showing how to use the pbrBERT components with the full nBERT model:

### Setting Up the Complete Pipeline

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertConfig

# Import pbrBERT components
from clinical_transformer.pt.training.pbrBERT.tokenizer import GeneExpressionTokenizer
from clinical_transformer.pt.training.pbrBERT.dataset import MaskedTokenDataset
from clinical_transformer.pt.training.pbrBERT.modeling import nBERTPretrainedModel, nBERTModelOutput

# Step 1: Prepare realistic gene expression data
def create_clinical_dataset():
    """Create a realistic clinical gene expression dataset"""
    
    # Common cancer-related genes
    gene_names = [
        "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "PIK3CA", "PTEN", "RB1", 
        "APC", "ATM", "CDH1", "CDKN2A", "MLH1", "MSH2", "VHL", "BRAF",
        "ERBB2", "MYC", "CCND1", "CDKN1A", "MDM2", "CHEK2", "PALB2", "RAD51"
    ]
    
    # Simulate different cancer subtypes with distinct expression patterns
    samples = []
    for sample_id in range(100):
        sample = {}
        
        # Create different expression patterns for different "subtypes"
        if sample_id < 30:  # High proliferation subtype
            high_genes = ["MYC", "CCND1", "EGFR", "ERBB2"]
            low_genes = ["CDKN1A", "CDKN2A", "RB1"]
        elif sample_id < 60:  # DNA repair deficient subtype  
            high_genes = ["ATM", "CHEK2", "RAD51", "PALB2"]
            low_genes = ["BRCA1", "BRCA2", "MLH1", "MSH2"]
        else:  # Apoptosis resistant subtype
            high_genes = ["MDM2", "PIK3CA", "BRAF"]
            low_genes = ["TP53", "PTEN", "VHL"]
        
        for gene in gene_names:
            base_expr = 5.0 + torch.randn(1).item() * 1.5  # Base expression with noise
            
            if gene in high_genes:
                expr = base_expr + 3.0 + torch.randn(1).item() * 0.5  # Higher expression
            elif gene in low_genes:
                expr = max(0.0, base_expr - 3.0 + torch.randn(1).item() * 0.5)  # Lower expression
            else:
                expr = base_expr + torch.randn(1).item() * 1.0  # Normal variation
            
            # Simulate some completely unexpressed genes
            if torch.rand(1).item() < 0.1:  # 10% chance of zero expression
                expr = 0.0
                
            sample[gene] = max(0.0, expr)  # Ensure non-negative
        
        samples.append(sample)
    
    return gene_names, samples

# Step 2: Initialize the complete pipeline
print("🧬 Setting up pbrBERT pipeline...")

gene_names, samples = create_clinical_dataset()
print(f"Created {len(samples)} samples with {len(gene_names)} genes")

# Initialize tokenizer
tokenizer = GeneExpressionTokenizer(gene_vocabulary=gene_names)
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

# Step 3: Create model configuration and model
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=1024,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pad_token_id=tokenizer.get_special_token_id('<pad>'),
    position_embedding_type="absolute",
    use_cache=True,
    classifier_dropout=None,
)

print(f"Model config: {config.hidden_size}D, {config.num_hidden_layers} layers, {config.num_attention_heads} heads")

# Initialize the nBERT model
model = nBERTPretrainedModel(config)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Step 4: Prepare data for training
print("\n📊 Preparing training data...")

# Tokenize the data
batch_encoding = tokenizer(samples, return_tensors=None)
tokens_list = batch_encoding['input_ids']
values_list = batch_encoding['gene_values']

print(f"Tokenized {len(tokens_list)} samples")
seq_lengths = [len(tokens) for tokens in tokens_list]
print(f"Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.1f}")

# Create dataset
dataset = MaskedTokenDataset(
    tokens=tokens_list,
    values=values_list,
    context_window=16,     # Reasonable context window for gene expression
    return_cls=True,       # Add CLS token for downstream tasks
    masking_fraction=0.15, # Standard BERT masking
    return_values=True
)

print(f"Dataset created: {len(dataset)} samples, output length: {dataset.context_window + 1}")

# Step 5: Create advanced collate function
def clinical_collate_fn(batch):
    """Advanced collate function for clinical data"""
    
    tokens = torch.stack([item['tokens'] for item in batch])
    values = torch.stack([item['values'] for item in batch])
    original_tokens = torch.stack([item['original_tokens'] for item in batch])
    
    # Create attention mask (True for real tokens, False for padding)
    attention_mask = (tokens != 0)
    
    # Create labels for MLM loss (-100 for ignored positions)
    labels = original_tokens.clone()
    labels[tokens == original_tokens] = -100  # Don't predict non-masked tokens
    labels[tokens == 0] = -100  # Don't predict padding tokens
    
    return {
        'tokens': tokens,
        'values': values,
        'attention_mask': attention_mask,
        'labels': labels,
        'original_tokens': original_tokens
    }

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=clinical_collate_fn,
    drop_last=True,
    num_workers=0  # Set to 0 for debugging, increase for production
)

print(f"DataLoader created: {len(dataloader)} batches of size 8")

# Step 6: Training loop with nBERT model
print("\n🚀 Training example...")

# Setup training
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Training loop
total_loss = 0
num_batches = 0

for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 5:  # Just show first 5 batches for demo
        break
    
    tokens = batch['tokens']
    values = batch['values']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    
    # Forward pass through nBERT
    outputs = model(
        tokens=tokens,
        values=values,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True
    )
    
    # Get predictions
    token_predictions = outputs.token_predictions  # Shape: [batch_size, seq_len, vocab_size]
    last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
    
    # Calculate masked language modeling loss
    # Reshape for loss calculation
    predictions_flat = token_predictions.view(-1, token_predictions.size(-1))
    labels_flat = labels.view(-1)
    
    loss = criterion(predictions_flat, labels_flat)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    num_batches += 1
    
    # Print detailed information for first batch
    if batch_idx == 0:
        print(f"\n📋 Detailed analysis of batch {batch_idx + 1}:")
        print(f"  Input shapes:")
        print(f"    tokens: {tokens.shape}")
        print(f"    values: {values.shape}")
        print(f"    attention_mask: {attention_mask.shape}")
        
        print(f"  Output shapes:")
        print(f"    last_hidden_state: {last_hidden_state.shape}")
        print(f"    token_predictions: {token_predictions.shape}")
        print(f"    num_layers: {len(outputs.hidden_states) if outputs.hidden_states else 'N/A'}")
        print(f"    num_attention_heads: {len(outputs.attentions) if outputs.attentions else 'N/A'}")
        
        # Show sample predictions
        sample_idx = 0
        sample_tokens = tokens[sample_idx]
        sample_labels = labels[sample_idx]
        sample_preds = token_predictions[sample_idx]
        
        print(f"  Sample sequence analysis:")
        masked_positions = (sample_labels != -100).nonzero().flatten()
        
        for pos in masked_positions[:3]:  # Show first 3 masked positions
            true_token = sample_labels[pos].item()
            pred_logits = sample_preds[pos]
            pred_token = pred_logits.argmax().item()
            confidence = torch.softmax(pred_logits, dim=0)[pred_token].item()
            
            true_gene = tokenizer.convert_ids_to_tokens(true_token)
            pred_gene = tokenizer.convert_ids_to_tokens(pred_token)
            
            print(f"    Position {pos.item()}: {true_gene} -> {pred_gene} (conf: {confidence:.3f})")
    
    print(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}")

avg_loss = total_loss / num_batches
print(f"\n📊 Training summary:")
print(f"  Batches processed: {num_batches}")
print(f"  Average loss: {avg_loss:.4f}")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Step 7: Inference example
print("\n🔮 Inference example...")

model.eval()
with torch.no_grad():
    # Take a sample from the dataset
    sample = dataset[0]
    tokens = sample['tokens'].unsqueeze(0)  # Add batch dimension
    values = sample['values'].unsqueeze(0)
    original_tokens = sample['original_tokens'].unsqueeze(0)
    
    # Run inference
    outputs = model(
        tokens=tokens,
        values=values,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True
    )
    
    # Analyze the results
    print(f"Inference results:")
    print(f"  Input sequence length: {tokens.shape[1]}")
    print(f"  Hidden state shape: {outputs.last_hidden_state.shape}")
    print(f"  Number of layers: {len(outputs.hidden_states)}")
    
    # Show attention patterns
    if outputs.attentions:
        attention = outputs.attentions[-1][0]  # Last layer, first sample
        print(f"  Attention shape (last layer): {attention.shape}")  # [num_heads, seq_len, seq_len]
        
        # Average attention across heads
        avg_attention = attention.mean(dim=0)  # [seq_len, seq_len]
        
        # Show attention to CLS token (position 0)
        cls_attention = avg_attention[0, 1:6]  # First 5 non-CLS tokens
        print(f"  CLS token attention to first 5 genes: {cls_attention.tolist()}")
    
    # Decode the sequence
    print(f"\n🧬 Sequence analysis:")
    for i in range(min(10, tokens.shape[1])):  # Show first 10 tokens
        token_id = tokens[0, i].item()
        value = values[0, i].item()
        
        if token_id == 0:  # Padding
            break
        elif token_id == 2:  # CLS
            print(f"  Position {i}: <CLS> (value: {value:.3f})")
        elif token_id == 1:  # MASK
            # Get prediction
            pred_logits = outputs.token_predictions[0, i]
            pred_token = pred_logits.argmax().item()
            confidence = torch.softmax(pred_logits, dim=0)[pred_token].item()
            pred_gene = tokenizer.convert_ids_to_tokens(pred_token)
            
            original_token = original_tokens[0, i].item()
            original_gene = tokenizer.convert_ids_to_tokens(original_token)
            
            print(f"  Position {i}: <MASK> -> {pred_gene} (was: {original_gene}, conf: {confidence:.3f}, value: {value:.3f})")
        else:
            gene_name = tokenizer.convert_ids_to_tokens(token_id)
            print(f"  Position {i}: {gene_name} (value: {value:.3f})")

print("\n✅ Complete nBERT pipeline example finished!")
print("\n🎯 Key takeaways:")
print("  • Gene expression data is tokenized by expression level ranking")
print("  • Values represent normalized expression positions (0-1)")
print("  • nBERT model outputs HuggingFace-compatible format")
print("  • Attention weights show gene-gene relationships")
print("  • Model can predict masked genes based on expression context")
```

## Multi-GPU Training with Accelerate

For large-scale training across multiple GPUs, here's how to use HuggingFace Accelerate:

### Installation

```bash
pip install accelerate
accelerate config  # Run this to configure your multi-GPU setup
```

### Multi-GPU Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers.models.bert.modeling_bert import BertConfig
from tqdm.auto import tqdm

# Import pbrBERT components
from clinical_transformer.pt.training.pbrBERT.tokenizer import GeneExpressionTokenizer
from clinical_transformer.pt.training.pbrBERT.dataset import MaskedTokenDataset
from clinical_transformer.pt.training.pbrBERT.modeling import nBERTPretrainedModel

def create_large_clinical_dataset():
    """Create a larger dataset for multi-GPU training"""
    
    # Extended gene list for more realistic training
    gene_names = [
        "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "PIK3CA", "PTEN", "RB1", 
        "APC", "ATM", "CDH1", "CDKN2A", "MLH1", "MSH2", "VHL", "BRAF",
        "ERBB2", "MYC", "CCND1", "CDKN1A", "MDM2", "CHEK2", "PALB2", "RAD51",
        "CTNNB1", "FBXW7", "NRAS", "PIK3R1", "SMAD4", "STK11", "NOTCH1", 
        "FAT1", "KEAP1", "NFE2L2", "RBM10", "TSC1", "TSC2", "ARID1A"
    ]
    
    # Create larger dataset (1000 samples for meaningful multi-GPU training)
    samples = []
    for sample_id in range(1000):
        sample = {}
        
        # Create different expression patterns
        if sample_id < 300:  # High proliferation subtype
            high_genes = ["MYC", "CCND1", "EGFR", "ERBB2", "CDKN1A"]
            low_genes = ["CDKN2A", "RB1", "TP53"]
        elif sample_id < 600:  # DNA repair deficient subtype  
            high_genes = ["ATM", "CHEK2", "RAD51", "PALB2", "NOTCH1"]
            low_genes = ["BRCA1", "BRCA2", "MLH1", "MSH2"]
        else:  # Apoptosis resistant subtype
            high_genes = ["MDM2", "PIK3CA", "BRAF", "KRAS", "NRAS"]
            low_genes = ["TP53", "PTEN", "VHL", "STK11"]
        
        for gene in gene_names:
            base_expr = 5.0 + torch.randn(1).item() * 1.5
            
            if gene in high_genes:
                expr = base_expr + 3.0 + torch.randn(1).item() * 0.5
            elif gene in low_genes:
                expr = max(0.0, base_expr - 3.0 + torch.randn(1).item() * 0.5)
            else:
                expr = base_expr + torch.randn(1).item() * 1.0
            
            # 5% chance of zero expression
            if torch.rand(1).item() < 0.05:
                expr = 0.0
                
            sample[gene] = max(0.0, expr)
        
        samples.append(sample)
    
    return gene_names, samples

def train_multi_gpu():
    """Multi-GPU training function using Accelerate"""
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
        mixed_precision='fp16',         # Use mixed precision for faster training
        log_with='tensorboard',         # Optional: log to tensorboard
        project_dir='./logs'            # Optional: log directory
    )
    
    # Print setup info
    accelerator.print(f"🚀 Starting multi-GPU training on {accelerator.num_processes} GPUs")
    accelerator.print(f"📊 Process {accelerator.process_index} of {accelerator.num_processes}")
    
    # Create dataset
    gene_names, samples = create_large_clinical_dataset()
    accelerator.print(f"Created {len(samples)} samples with {len(gene_names)} genes")
    
    # Initialize tokenizer
    tokenizer = GeneExpressionTokenizer(gene_vocabulary=gene_names)
    
    # Create model configuration
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,      # Larger model for multi-GPU training
        num_hidden_layers=12, # Deeper network
        num_attention_heads=16,
        intermediate_size=2048,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=tokenizer.get_special_token_id('<pad>'),
        position_embedding_type="absolute",
    )
    
    # Initialize model
    model = nBERTPretrainedModel(config)
    accelerator.print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Tokenize data
    batch_encoding = tokenizer(samples, return_tensors=None)
    tokens_list = batch_encoding['input_ids']
    values_list = batch_encoding['gene_values']
    
    # Create dataset with larger context window
    dataset = MaskedTokenDataset(
        tokens=tokens_list,
        values=values_list,
        context_window=32,     # Larger context for better learning
        return_cls=True,
        masking_fraction=0.15,
        return_values=True
    )
    
    # Create DataLoader with larger batch size for multi-GPU
    def clinical_collate_fn(batch):
        tokens = torch.stack([item['tokens'] for item in batch])
        values = torch.stack([item['values'] for item in batch])
        original_tokens = torch.stack([item['original_tokens'] for item in batch])
        
        attention_mask = (tokens != 0)
        labels = original_tokens.clone()
        labels[tokens == original_tokens] = -100
        labels[tokens == 0] = -100
        
        return {
            'tokens': tokens,
            'values': values,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Larger batch size per GPU
        shuffle=True,
        collate_fn=clinical_collate_fn,
        drop_last=True,
        num_workers=4  # More workers for faster data loading
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-5,           # Learning rate for multi-GPU
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Learning rate scheduler
    num_training_steps = len(dataloader) * 3  # 3 epochs
    num_warmup_steps = num_training_steps // 10  # 10% warmup
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare everything with accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # Training loop
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    accelerator.print(f"🏋️ Starting training for 3 epochs")
    accelerator.print(f"📦 Batch size per GPU: 16")
    accelerator.print(f"📦 Total batch size: {16 * accelerator.num_processes}")
    accelerator.print(f"📦 Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
    accelerator.print(f"📦 Total optimization steps: {num_training_steps}")
    
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(3):
        total_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                tokens = batch['tokens']
                values = batch['values']
                labels = batch['labels']
                
                # Forward pass
                outputs = model(
                    tokens=tokens,
                    values=values,
                    return_dict=True
                )
                
                # Calculate loss
                token_predictions = outputs.token_predictions
                predictions_flat = token_predictions.view(-1, token_predictions.size(-1))
                labels_flat = labels.view(-1)
                
                loss = criterion(predictions_flat, labels_flat)
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update progress
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Log every 50 steps
                if step % 50 == 0 and accelerator.is_main_process:
                    current_lr = scheduler.get_last_lr()[0]
                    avg_loss = total_loss / num_batches
                    accelerator.print(
                        f"Epoch {epoch+1}, Step {step}: "
                        f"Loss = {loss.item():.4f}, "
                        f"Avg Loss = {avg_loss:.4f}, "
                        f"LR = {current_lr:.2e}"
                    )
        
        # End of epoch summary
        avg_loss = total_loss / num_batches
        accelerator.print(f"✅ Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if accelerator.is_main_process:
            accelerator.save_state(f'./checkpoints/epoch_{epoch+1}')
            accelerator.print(f"💾 Checkpoint saved for epoch {epoch+1}")
    
    # Final model save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained('./final_model')
        tokenizer.save_pretrained('./final_model')
        accelerator.print("🎉 Training completed! Final model saved.")

# Example usage
if __name__ == "__main__":
    train_multi_gpu()
```

### Running Multi-GPU Training

```bash
# First, configure accelerate
accelerate config

# Then run the training script
accelerate launch train_multi_gpu.py

# Or specify specific configuration
accelerate launch --config_file config.yaml train_multi_gpu.py

# For debugging on single GPU
accelerate launch --num_processes=1 train_multi_gpu.py
```

### Accelerate Configuration Example

When you run `accelerate config`, you'll be prompted for settings. Here's an example configuration:

```yaml
# config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4  # Number of GPUs
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### Key Benefits of Multi-GPU Training

1. **Faster Training**: Distribute batches across multiple GPUs
2. **Larger Models**: Train bigger models that don't fit on single GPU
3. **Mixed Precision**: Use FP16 for faster training and reduced memory
4. **Gradient Accumulation**: Simulate larger batch sizes
5. **Automatic Synchronization**: Accelerate handles device placement and gradient sync

### Performance Tips

- **Batch Size**: Use `batch_size_per_gpu * num_gpus` for total batch size
- **Learning Rate**: Scale learning rate with total batch size
- **Data Loading**: Use more `num_workers` for faster data loading
- **Memory**: Monitor GPU memory usage and adjust batch size accordingly
- **Checkpointing**: Save checkpoints regularly for long training runs

```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{pbrbert2025,
  title={pbrBERT: Gene Expression Tokenizer and Dataset for Clinical Transformers},
  author={Clinical Transformer Team},
  year={2025},
  url={https://github.com/<your-org>/clinical_transformer}
}
```
