"""
Quick check of current training status
"""

import torch
import os
from datetime import datetime

# Check latest checkpoint
checkpoint_path = "checkpoints/last.pth"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("="*60)
    print("LATEST CHECKPOINT STATUS")
    print("="*60)
    
    # File modification time
    mod_time = os.path.getmtime(checkpoint_path)
    mod_datetime = datetime.fromtimestamp(mod_time)
    print(f"Last updated: {mod_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Checkpoint contents
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    
    if 'best_acc' in checkpoint:
        print(f"\nBest accuracy: {checkpoint['best_acc']}")
    
    if 'timestamp' in checkpoint:
        ts = checkpoint['timestamp']
        ts_datetime = datetime.fromtimestamp(ts)
        print(f"Checkpoint timestamp: {ts_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*60)
    
    # Try to determine if using local or remote data
    # by checking if there are recent validation metrics
    print("\nTo verify data source, check training output for:")
    print("  - File paths being loaded")
    print("  - Validation loss values")
    print("  - Any error messages about data loading")
else:
    print("No checkpoint found yet")

# Check latest training metrics
import glob
metric_files = glob.glob("stats/*/20251125_220600/training_metrics.csv")
if metric_files:
    latest_metrics = metric_files[0]
    print(f"\n{'='*60}")
    print(f"LATEST TRAINING METRICS")
    print(f"{'='*60}")
    print(f"File: {latest_metrics}")
    
    with open(latest_metrics, 'r') as f:
        lines = f.readlines()
        print(f"\nTotal lines: {len(lines)}")
        if len(lines) > 1:
            print("\nLast 5 entries:")
            for line in lines[-5:]:
                print(f"  {line.strip()}")
