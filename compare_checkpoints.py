"""
Compare checkpoints from distributed training vs simple training.
"""

import torch
import os


def load_and_inspect_checkpoint(checkpoint_path, name):
    """Load and display checkpoint information."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found!")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"\n📦 Checkpoint Contents:")
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                if key not in ['model_state_dict', 'optimizer_state_dict', 'state_dict']:
                    print(f"  - {key}: {checkpoint[key]}")
        
        # Extract metrics
        metrics = {}
        if isinstance(checkpoint, dict):
            # Try different key formats
            metrics['epoch'] = checkpoint.get('epoch', 'N/A')
            
            # Training metrics
            metrics['train_loss'] = checkpoint.get('train_loss', checkpoint.get('loss', 'N/A'))
            metrics['train_acc'] = checkpoint.get('train_acc', checkpoint.get('accuracy', 'N/A'))
            
            # Validation metrics
            metrics['val_loss'] = checkpoint.get('val_loss', 'N/A')
            metrics['val_acc'] = checkpoint.get('val_acc', checkpoint.get('best_acc', checkpoint.get('best_accuracy', 'N/A')))
        
        print(f"\n📊 Metrics:")
        print(f"  Epoch: {metrics['epoch']}")
        print(f"  Train Loss: {metrics['train_loss']}")
        print(f"  Train Acc: {metrics['train_acc']}")
        print(f"  Val Loss: {metrics['val_loss']}")
        print(f"  Val Acc: {metrics['val_acc']}")
        
        # Check for issues
        print(f"\n🔍 Health Check:")
        
        if isinstance(metrics['val_loss'], (int, float)):
            if metrics['val_loss'] > 1000:
                print(f"  ⚠️  CRITICAL: Validation loss is extremely high ({metrics['val_loss']:.2f})")
                print(f"     This indicates a serious problem with validation data or model")
            elif metrics['val_loss'] > 10:
                print(f"  ⚠️  WARNING: Validation loss is high ({metrics['val_loss']:.2f})")
            else:
                print(f"  ✅ Validation loss is reasonable ({metrics['val_loss']:.4f})")
        
        if isinstance(metrics['val_acc'], (int, float)):
            if metrics['val_acc'] < 1:
                print(f"  ⚠️  CRITICAL: Validation accuracy is extremely low ({metrics['val_acc']:.2f}%)")
                print(f"     Model is performing worse than random guessing")
            elif metrics['val_acc'] < 5:
                print(f"  ⚠️  WARNING: Validation accuracy is low ({metrics['val_acc']:.2f}%)")
            else:
                print(f"  ✅ Validation accuracy is reasonable ({metrics['val_acc']:.2f}%)")
        
        # Check model weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Check for NaN/Inf in weights
        has_nan = False
        has_inf = False
        for name, param in state_dict.items():
            if torch.isnan(param).any():
                has_nan = True
            if torch.isinf(param).any():
                has_inf = True
        
        if has_nan:
            print(f"  ⚠️  CRITICAL: Model weights contain NaN values!")
        if has_inf:
            print(f"  ⚠️  CRITICAL: Model weights contain Inf values!")
        if not has_nan and not has_inf:
            print(f"  ✅ Model weights are healthy (no NaN/Inf)")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_checkpoints():
    """Compare distributed vs simple training checkpoints."""
    
    print("\n" + "="*60)
    print("CHECKPOINT COMPARISON")
    print("="*60)
    
    # Load distributed training checkpoint
    distributed_metrics = load_and_inspect_checkpoint(
        "checkpoints/best.pth",
        "DISTRIBUTED TRAINING (WebDataset)"
    )
    
    # Load simple training checkpoint
    simple_metrics = load_and_inspect_checkpoint(
        "checkpoints_simple/best.pth",
        "SIMPLE TRAINING (ImageFolder)"
    )
    
    # Comparison
    if distributed_metrics and simple_metrics:
        print(f"\n{'='*60}")
        print("SIDE-BY-SIDE COMPARISON")
        print(f"{'='*60}")
        
        print(f"\n{'Metric':<20} {'Distributed':<20} {'Simple':<20} {'Difference':<20}")
        print("-" * 80)
        
        # Epoch
        print(f"{'Epoch':<20} {str(distributed_metrics['epoch']):<20} {str(simple_metrics['epoch']):<20} {'-':<20}")
        
        # Training Loss
        if isinstance(distributed_metrics['train_loss'], (int, float)) and isinstance(simple_metrics['train_loss'], (int, float)):
            diff = distributed_metrics['train_loss'] - simple_metrics['train_loss']
            print(f"{'Train Loss':<20} {distributed_metrics['train_loss']:<20.4f} {simple_metrics['train_loss']:<20.4f} {diff:<20.4f}")
        
        # Training Accuracy
        if isinstance(distributed_metrics['train_acc'], (int, float)) and isinstance(simple_metrics['train_acc'], (int, float)):
            diff = distributed_metrics['train_acc'] - simple_metrics['train_acc']
            print(f"{'Train Acc (%)':<20} {distributed_metrics['train_acc']:<20.2f} {simple_metrics['train_acc']:<20.2f} {diff:<20.2f}")
        
        # Validation Loss
        if isinstance(distributed_metrics['val_loss'], (int, float)) and isinstance(simple_metrics['val_loss'], (int, float)):
            diff = distributed_metrics['val_loss'] - simple_metrics['val_loss']
            ratio = distributed_metrics['val_loss'] / simple_metrics['val_loss'] if simple_metrics['val_loss'] != 0 else float('inf')
            print(f"{'Val Loss':<20} {distributed_metrics['val_loss']:<20.4f} {simple_metrics['val_loss']:<20.4f} {diff:<20.4f}")
            if ratio > 1000:
                print(f"  ⚠️  CRITICAL: Distributed val loss is {ratio:.0f}x higher than simple!")
        
        # Validation Accuracy
        if isinstance(distributed_metrics['val_acc'], (int, float)) and isinstance(simple_metrics['val_acc'], (int, float)):
            diff = distributed_metrics['val_acc'] - simple_metrics['val_acc']
            print(f"{'Val Acc (%)':<20} {distributed_metrics['val_acc']:<20.2f} {simple_metrics['val_acc']:<20.2f} {diff:<20.2f}")
        
        # Analysis
        print(f"\n{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")
        
        if isinstance(distributed_metrics['val_loss'], (int, float)) and isinstance(simple_metrics['val_loss'], (int, float)):
            if distributed_metrics['val_loss'] > 1000 and simple_metrics['val_loss'] < 10:
                print("\n🔴 CRITICAL ISSUE DETECTED:")
                print("  - Simple training: Validation loss is normal")
                print("  - Distributed training: Validation loss is astronomical")
                print("\n  This suggests the distributed training is STILL using incorrect validation data.")
                print("  Possible causes:")
                print("    1. local_simulation.py changes not applied")
                print("    2. Cached/old process still running")
                print("    3. Different code path for validation in distributed mode")
                print("\n  Recommended actions:")
                print("    1. Stop current training process")
                print("    2. Verify local_simulation.py has correct paths")
                print("    3. Clear any cached data")
                print("    4. Restart training from scratch")
            elif abs(distributed_metrics['val_loss'] - simple_metrics['val_loss']) < 2:
                print("\n✅ VALIDATION LOOKS GOOD:")
                print("  Both training methods have similar validation loss")
                print("  This indicates the data pipeline is working correctly")


if __name__ == "__main__":
    compare_checkpoints()
