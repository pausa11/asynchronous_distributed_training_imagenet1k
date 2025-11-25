"""
Script to test the trained model and diagnose validation issues.
This script loads a checkpoint and validates it on the test set,
providing detailed diagnostics about the preprocessing pipeline.
"""

import torch
import torch.nn as nn
from src.model import get_model
from src.dataset import get_imagenet_dataset, make_transform
from src.utils import get_device
import argparse
from PIL import Image
import numpy as np


def test_preprocessing_pipeline():
    """Test that preprocessing is working correctly."""
    print("\n" + "="*60)
    print("TESTING PREPROCESSING PIPELINE")
    print("="*60)
    
    # Create both transforms
    train_transform = make_transform(train=True)
    val_transform = make_transform(train=False)
    
    print("\nTraining Transform:")
    print(train_transform)
    print("\nValidation Transform:")
    print(val_transform)
    
    # Check if both have normalization
    train_has_norm = any('Normalize' in str(t) for t in train_transform.transforms)
    val_has_norm = any('Normalize' in str(t) for t in val_transform.transforms)
    
    print(f"\n✓ Training has normalization: {train_has_norm}")
    print(f"✓ Validation has normalization: {val_has_norm}")
    
    if not val_has_norm:
        print("\n⚠️  WARNING: Validation transform is missing normalization!")
        print("   This would cause extremely high validation loss.")
        return False
    
    return True


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    model = get_model(num_classes=200).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"  Best accuracy: {checkpoint.get('best_accuracy', 'unknown')}")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume it's just the state dict
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        print("✓ Checkpoint loaded successfully")
        return model
    
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        raise


def validate_model(model, val_loader, device, max_batches=None):
    """Validate the model and return detailed metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total_samples = 0
    num_batches = 0
    
    # Track predictions for analysis
    all_predictions = []
    all_targets = []
    all_losses = []
    
    print("\n" + "="*60)
    print("RUNNING VALIDATION")
    print("="*60)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break
            
            data, target = data.to(device), target.to(device)
            
            # Check for NaN or extreme values in input
            if torch.isnan(data).any():
                print(f"⚠️  WARNING: NaN detected in batch {i} input data")
            if torch.isinf(data).any():
                print(f"⚠️  WARNING: Inf detected in batch {i} input data")
            
            # Print statistics about input data
            if i == 0:
                print(f"\nInput data statistics (first batch):")
                print(f"  Shape: {data.shape}")
                print(f"  Min: {data.min().item():.4f}")
                print(f"  Max: {data.max().item():.4f}")
                print(f"  Mean: {data.mean().item():.4f}")
                print(f"  Std: {data.std().item():.4f}")
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Check for NaN in output
            if torch.isnan(output).any():
                print(f"⚠️  WARNING: NaN detected in batch {i} model output")
            if torch.isnan(loss):
                print(f"⚠️  WARNING: NaN loss in batch {i}")
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct_batch = pred.eq(target.view_as(pred)).sum().item()
            
            correct += correct_batch
            total_samples += len(target)
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
            all_losses.append(loss.item())
            
            if i % 10 == 0:
                batch_acc = 100. * correct_batch / len(target)
                print(f"Batch {i:3d}: Loss={loss.item():8.4f}, Acc={batch_acc:6.2f}%")
    
    # Calculate final metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_acc = 100. * correct / total_samples if total_samples > 0 else 0
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Total batches: {num_batches}")
    print(f"Total samples: {total_samples}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Accuracy: {avg_acc:.2f}%")
    print(f"Correct predictions: {correct}/{total_samples}")
    
    # Additional diagnostics
    print(f"\nLoss statistics:")
    print(f"  Min loss: {min(all_losses):.4f}")
    print(f"  Max loss: {max(all_losses):.4f}")
    print(f"  Std loss: {np.std(all_losses):.4f}")
    
    # Check prediction distribution
    unique_preds = len(set(all_predictions))
    print(f"\nPrediction diversity:")
    print(f"  Unique predictions: {unique_preds}/200 classes")
    if unique_preds < 10:
        print(f"  ⚠️  WARNING: Model is only predicting {unique_preds} classes!")
        print(f"     This suggests the model may have collapsed.")
    
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description='Test trained model on validation set')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--val_dataset_url', type=str, 
                        default='file:data/tiny-imagenet-wds/val-{0000..0009}.tar',
                        help='Validation dataset URL pattern')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for validation')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to test (for quick tests)')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Test preprocessing pipeline first
    print("\n" + "="*60)
    print("STEP 1: VERIFY PREPROCESSING PIPELINE")
    print("="*60)
    preprocessing_ok = test_preprocessing_pipeline()
    
    if not preprocessing_ok:
        print("\n❌ CRITICAL: Preprocessing pipeline has issues!")
        print("   Please fix the validation transform before proceeding.")
        return
    
    # Load checkpoint
    print("\n" + "="*60)
    print("STEP 2: LOAD MODEL CHECKPOINT")
    print("="*60)
    model = load_checkpoint(args.checkpoint, device)
    
    # Load validation dataset
    print("\n" + "="*60)
    print("STEP 3: LOAD VALIDATION DATASET")
    print("="*60)
    print(f"Loading validation data from: {args.val_dataset_url}")
    val_loader = get_imagenet_dataset(
        args.val_dataset_url, 
        batch_size=args.batch_size, 
        num_workers=0,  # Avoid multiprocessing issues
        train=False  # Use validation transforms
    )
    print("✓ Validation dataset loaded")
    
    # Run validation
    print("\n" + "="*60)
    print("STEP 4: VALIDATE MODEL")
    print("="*60)
    val_loss, val_acc = validate_model(model, val_loader, device, args.max_batches)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    
    if val_loss > 10:
        print("\n⚠️  WARNING: Very high validation loss detected!")
        print("   Possible causes:")
        print("   1. Missing normalization in validation (check preprocessing)")
        print("   2. Model not properly trained")
        print("   3. Data corruption or mismatch")
    
    if val_acc < 1:
        print("\n⚠️  WARNING: Very low accuracy!")
        print("   The model appears to be performing worse than random guessing.")


if __name__ == "__main__":
    main()
