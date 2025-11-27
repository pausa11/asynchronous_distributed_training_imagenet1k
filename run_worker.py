#!/usr/bin/env python
"""
Wrapper script to run a Worker.
This ensures the module is imported correctly for RPC.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rpc_worker import run_worker

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=3)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29500")
    parser.add_argument("--dataset_url", type=str, required=True)
    parser.add_argument("--val_dataset_url", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_webdataset", action="store_true")
    args = parser.parse_args()
    
    run_worker(args.rank, args.world_size, args.master_addr, args.master_port, 
               args.dataset_url, args.val_dataset_url, args.epochs, args.use_webdataset)
