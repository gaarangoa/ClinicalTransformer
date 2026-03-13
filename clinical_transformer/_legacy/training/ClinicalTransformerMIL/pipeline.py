import os
from tqdm import tqdm
import h5py

from clinical_transformer._legacy.losses.contrastive import ntxent as contrastive_loss
from clinical_transformer._legacy.training.ClinicalTransformerMIL.dataset import MILDataset
from clinical_transformer import ClinicalTransformerMILConfig, ClinicalTransformerMILModel
from torch.utils.data import DataLoader
import torch.distributed as dist

from accelerate import Accelerator
import torch
from pathlib import Path
import csv

import torch.multiprocessing as mp

def load_data(rank, world_size):
    # now load your dataset
    embeddings_path = '/path/to/your/embeddings/'
    sample_filenames = os.listdir(embeddings_path)

    total_samples = len(sample_filenames)
    samples_per_rank = total_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank != world_size-1 else total_samples

    data = []
    for sample_filename in tqdm(sample_filenames[start_idx: end_idx]):
        with h5py.File(f'{embeddings_path}/{sample_filename}', 'r') as f:
            embeddings = torch.tensor(f['embeddings'][:])[0]
            data.append({
                'sample_id': sample_filename.split('.')[-1],
                'embeddings': embeddings.to(torch.bfloat16),
            })
    
    dataset = MILDataset(data, max_length=10000, random_mask_size=8000)

    return dataset

def run(from_pretrained=None):
    # Dataset
    save_dir = Path('../../models/CT-value-based-RNA-MIL/')
    save_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="bf16")
    
    csv_path = os.path.join(save_dir, "metrics.csv")
    # Create CSV and write header once before training (only on main process)
    if accelerator.is_main_process:
        with open(csv_path, 'w', newline='') as fo:
            writer = csv.writer(fo)
            writer.writerow(["epoch", "batch", "loss"])  # header

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    dataset = load_data(rank, world_size)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=135, 
        shuffle=True, 
        num_workers=12, 
        pin_memory=True, 
        persistent_workers=False,
        drop_last=False,
    )
    
    # Model
    save_interval = 100
    
    config = ClinicalTransformerMILConfig()
    model = ClinicalTransformerMILModel(config)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model, optimizer = accelerator.prepare(model, optimizer)

    for epoch in range(1000):
        
        for batch_idx, (tokens, mask, randmask) in enumerate(dataloader):
            optimizer.zero_grad()
            with accelerator.autocast():
                out1 = model(tokens, mask=mask, return_attentions=False)
                out2 = model(tokens, mask=randmask, return_attentions=False)            
                loss = contrastive_loss(out1.hidden_states, out2.hidden_states)
        
            accelerator.backward(loss)
            optimizer.step()

            msg = f"Epoch: {epoch} - Step: {batch_idx} - Loss: {loss.item()}"
            accelerator.print("\r" + msg.ljust(100), end="", flush=True)

            with open(csv_path, 'a', newline='') as fo:
                writer = csv.writer(fo)
                writer.writerow([epoch, batch_idx, loss.item()])
        
        if (epoch + 1) % save_interval == 0 and accelerator.is_main_process:
            model_to_save = accelerator.unwrap_model(model)
            hf_save_dir = os.path.join(save_dir, f"epoch{epoch+1}")
            os.makedirs(hf_save_dir, exist_ok=True)
            model_to_save.save_pretrained(hf_save_dir)
            accelerator.print(f"Saved model with save_pretrained to: {hf_save_dir}")
        
        
if __name__ == "__main__":
    from_pretrained = ''
    run(from_pretrained)
