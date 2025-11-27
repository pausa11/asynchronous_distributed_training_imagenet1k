"""
Diagnostic script to compare PS weights with a fresh model.
This helps identify if the PS weights are corrupted.
"""

import torch
from src.model import get_model
from src.rpc_ps import ParameterServer
import os

def check_weights():
    print("="*60)
    print("DIAGNOSTIC: Checking Parameter Server Weights")
    print("="*60)
    
    # Load checkpoint if exists
    checkpoint_path = "checkpoints/last.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"\n✓ Found checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check for NaN or Inf in weights
        has_nan = False
        has_inf = False
        weight_stats = []
        
        for name, param in checkpoint['model_state_dict'].items():
            # Skip non-floating point tensors (like indices)
            if not param.dtype.is_floating_point:
                continue
                
            if torch.isnan(param).any():
                has_nan = True
                print(f"❌ NaN found in: {name}")
            if torch.isinf(param).any():
                has_inf = True
                print(f"❌ Inf found in: {name}")
            
            weight_stats.append({
                'name': name,
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item()
            })
        
        if has_nan or has_inf:
            print("\n❌ PROBLEM FOUND: Weights contain NaN or Inf values!")
            print("   This explains the terrible validation loss.")
            print("   Solution: Delete checkpoints and restart training.")
        else:
            print("\n✓ No NaN or Inf values found in weights")
            
            # Show some weight statistics
            print("\nWeight Statistics (first 5 layers):")
            for stat in weight_stats[:5]:
                print(f"  {stat['name'][:40]:40s} | mean: {stat['mean']:8.4f} | std: {stat['std']:8.4f} | min: {stat['min']:8.4f} | max: {stat['max']:8.4f}")
        
        # Check optimizer state
        if 'optimizer_state_dict' in checkpoint:
            print("\n✓ Optimizer state found")
            print(f"  Best accuracy: {checkpoint.get('best_acc', 0.0):.2f}%")
    else:
        print(f"\n❌ No checkpoint found at: {checkpoint_path}")
        print("   This is normal if training just started.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_weights()
