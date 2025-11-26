import torch.multiprocessing as mp
import os
import time
from src.rpc_ps import run_ps
from src.rpc_worker import run_worker

def run_simulation():
    world_size = 2
    master_addr = "localhost"
    master_port = "30005"
    # Use local WebDataset files (verified to be correct)
    dataset_url = "file:data/tiny-imagenet-wds/train/train-{000000..000002}.tar"
    val_dataset_url = "file:data/tiny-imagenet-wds/val/val-{000000..000001}.tar"
    
    mp.set_start_method("spawn", force=True)
    
    checkpoint_dir = "checkpoints"
    p0 = mp.Process(target=run_ps, args=(0, world_size, master_addr, master_port, checkpoint_dir))
    p1 = mp.Process(target=run_worker, args=(1, world_size, master_addr, master_port, dataset_url, val_dataset_url, 5))
    
    p0.start()
    p1.start()
    
    p0.join()
    p1.join()

if __name__ == "__main__":
    run_simulation()
