import torch.multiprocessing as mp
import os
import time
from src.rpc_ps import run_ps
from src.rpc_worker import run_worker

def run_simulation():
    world_size = 2
    master_addr = "localhost"
    master_port = "30005"
    
    # Use UNCOMPRESSED dataset (like train_simple.py)
    # This is the local path to tiny-imagenet-200 directory
    dataset_url = "data/tiny-imagenet-200"
    val_dataset_url = "data/tiny-imagenet-200"  # Same root, will use train/ and val/ subdirs
    use_webdataset = False  # Use ImageFolder instead of WebDataset
    
    mp.set_start_method("spawn", force=True)
    
    checkpoint_dir = "checkpoints"
    p0 = mp.Process(target=run_ps, args=(0, world_size, master_addr, master_port, checkpoint_dir))
    p1 = mp.Process(target=run_worker, args=(1, world_size, master_addr, master_port, dataset_url, val_dataset_url, 5, use_webdataset))
    
    p0.start()
    p1.start()
    
    p0.join()
    p1.join()

if __name__ == "__main__":
    run_simulation()
