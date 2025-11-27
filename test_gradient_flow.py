"""
Test script to verify gradient flow in distributed training.
Compares a single training step between train_simple.py approach and distributed approach.
"""

import torch
import torch.nn as nn
from src.model import get_model
from src.dataset import get_imagenet_folder_dataset
from src.utils import get_device

def test_single_batch():
    print("="*60)
    print("TEST: Single Batch Training Comparison")
    print("="*60)
    
    device = get_device()
    
    # Create model and optimizer (matching train_simple.py)
    model = get_model(num_classes=200).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Get a single batch of data
    loader = get_imagenet_folder_dataset("data/tiny-imagenet-200", batch_size=64, num_workers=0, train=True)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)
    
    # Save initial weights
    initial_weight = model.fc.weight.data.clone()
    
    print(f"\n1. Initial fc layer weight mean: {initial_weight.mean().item():.6f}")
    
    # Forward pass
    output = model(data)
    loss = criterion(output, target)
    print(f"2. Loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradient
    grad_mean = model.fc.weight.grad.mean().item()
    grad_std = model.fc.weight.grad.std().item()
    print(f"3. Gradient mean: {grad_mean:.6f}, std: {grad_std:.6f}")
    
    # Optimizer step
    optimizer.step()
    
    # Check weight change
    final_weight = model.fc.weight.data
    weight_change = (final_weight - initial_weight).abs().mean().item()
    print(f"4. Weight change after step: {weight_change:.6f}")
    print(f"5. Final fc layer weight mean: {final_weight.mean().item():.6f}")
    
    # Forward pass again to see if loss decreased
    output2 = model(data)
    loss2 = criterion(output2, target)
    print(f"6. Loss after update: {loss2.item():.4f}")
    print(f"7. Loss change: {loss.item() - loss2.item():.6f}")
    
    if weight_change < 1e-6:
        print("\n❌ WARNING: Weights barely changed! Optimizer might not be working.")
    elif loss2.item() >= loss.item():
        print("\n⚠️  WARNING: Loss did not decrease. This is normal for a single step but concerning if persistent.")
    else:
        print("\n✓ Weights updated and loss decreased as expected.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_single_batch()
