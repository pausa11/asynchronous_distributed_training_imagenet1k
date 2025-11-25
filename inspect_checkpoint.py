"""
Inspect checkpoint to understand training state
"""

import torch

checkpoint_path = "checkpoints/best.pth"

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*60)
print("CHECKPOINT CONTENTS")
print("="*60)

if isinstance(checkpoint, dict):
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    print("\n" + "-"*60)
    
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    
    if 'best_accuracy' in checkpoint:
        print(f"Best Accuracy: {checkpoint['best_accuracy']}")
    
    if 'optimizer_state_dict' in checkpoint:
        print("Optimizer state: Present")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print("\n" + "-"*60)
    print("MODEL WEIGHTS STATISTICS")
    print("-"*60)
    
    # Check a few key layers
    sample_layers = []
    for name, param in list(state_dict.items())[:10]:
        if 'weight' in name or 'bias' in name:
            sample_layers.append((name, param))
    
    for name, param in sample_layers:
        print(f"\n{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Mean: {param.mean().item():.6f}")
        print(f"  Std: {param.std().item():.6f}")
        print(f"  Min: {param.min().item():.6f}")
        print(f"  Max: {param.max().item():.6f}")
        
        # Check for NaN or Inf
        if torch.isnan(param).any():
            print(f"  ⚠️  WARNING: Contains NaN values!")
        if torch.isinf(param).any():
            print(f"  ⚠️  WARNING: Contains Inf values!")
    
    # Check final layer (classifier)
    print("\n" + "-"*60)
    print("CLASSIFIER LAYER (FINAL LAYER)")
    print("-"*60)
    
    classifier_keys = [k for k in state_dict.keys() if 'fc' in k or 'classifier' in k or 'head' in k]
    for key in classifier_keys[-2:]:  # Last 2 keys (weight and bias)
        param = state_dict[key]
        print(f"\n{key}:")
        print(f"  Shape: {param.shape}")
        print(f"  Mean: {param.mean().item():.6f}")
        print(f"  Std: {param.std().item():.6f}")
        print(f"  Min: {param.min().item():.6f}")
        print(f"  Max: {param.max().item():.6f}")
        
        if torch.isnan(param).any():
            print(f"  ⚠️  WARNING: Contains NaN values!")
        if torch.isinf(param).any():
            print(f"  ⚠️  WARNING: Contains Inf values!")

else:
    print("Checkpoint is a state dict directly")
    state_dict = checkpoint

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

# Count total parameters
total_params = sum(p.numel() for p in state_dict.values())
print(f"Total parameters: {total_params:,}")

# Check for anomalies
has_nan = any(torch.isnan(p).any() for p in state_dict.values())
has_inf = any(torch.isinf(p).any() for p in state_dict.values())

print(f"Contains NaN: {has_nan}")
print(f"Contains Inf: {has_inf}")

if has_nan or has_inf:
    print("\n⚠️  CRITICAL: Model weights contain NaN or Inf values!")
    print("   This explains the extreme loss values.")
    print("   The training process likely had numerical instability.")
