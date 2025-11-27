"""
Direct test: Load checkpoint and run validation to see if the problem is in the checkpoint itself.
"""

import torch
import torch.nn as nn
from src.model import get_model
from src.dataset import get_imagenet_folder_dataset
from src.utils import get_device
import os

def test_checkpoint_validation():
    print("="*60)
    print("TEST: Validating with Checkpoint Weights")
    print("="*60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load validation data
    print("\nLoading validation data...")
    val_loader = get_imagenet_folder_dataset("data/tiny-imagenet-200", batch_size=64, num_workers=0, train=False)
    
    # Create model
    model = get_model(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Load checkpoint if exists
    checkpoint_path = "checkpoints/last.pth"
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Checkpoint loaded")
    else:
        print("\n⚠️  No checkpoint found, using random weights")
    
    # Put model in eval mode
    model.eval()
    print(f"Model mode: {'eval' if not model.training else 'train'}")
    
    # Run validation on first 5 batches
    print("\nRunning validation on first 5 batches...")
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            if batch_idx >= 5:  # Only test first 5 batches
                break
                
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ Batch {batch_idx}: Loss is NaN or Inf!")
                print(f"   Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
                
                # Check model weights
                for name, param in model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"   ❌ Parameter {name} contains NaN or Inf")
                break
            
            # Statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            batch_acc = 100. * correct / total
            print(f"Batch {batch_idx}: Loss={loss.item():.4f}, Acc={batch_acc:.2f}%")
    
    if total > 0:
        avg_loss = running_loss / min(5, batch_idx + 1)
        avg_acc = 100. * correct / total
        print(f"\nAverage over {min(5, batch_idx + 1)} batches:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {avg_acc:.2f}%")
        
        if avg_loss > 1000:
            print("\n❌ PROBLEM: Loss is extremely high!")
            print("   This suggests the model weights are corrupted.")
        else:
            print("\n✓ Loss looks reasonable")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_checkpoint_validation()
