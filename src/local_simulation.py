import torch.multiprocessing as mp
import os
import time
from src.rpc_ps import run_ps
from src.rpc_worker import run_worker

def run_simulation():
    world_size = 2
    master_addr = "localhost"
    master_port = "30004"
    # Use public HTTP URL to stream data from GCS bucket
    # This avoids gsutil authentication issues in multiprocessing and saves local disk space
    dataset_url = "https://storage.googleapis.com/caso-estudio-2/tiny-imagenet-wds/train/train-{000000..000002}.tar" 
    
    mp.set_start_method("spawn", force=True)
    
    checkpoint_dir = "checkpoints"
    p0 = mp.Process(target=run_ps, args=(0, world_size, master_addr, master_port, checkpoint_dir))
    p1 = mp.Process(target=run_worker, args=(1, world_size, master_addr, master_port, dataset_url, 5))
    
    p0.start()
    p1.start()
    
    p0.join()
    p1.join()

if __name__ == "__main__":
    run_simulation()
