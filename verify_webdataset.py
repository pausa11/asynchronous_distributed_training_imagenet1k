"""
Verify WebDataset integrity and compare with raw data.
"""

import webdataset as wds
import torch
from torchvision import transforms
from PIL import Image
import io
import tarfile
import numpy as np


def inspect_tar_file(tar_path, num_samples=5):
    """Inspect the contents of a tar file."""
    print(f"\nInspecting: {tar_path}")
    print("="*60)
    
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        print(f"Total files in tar: {len(members)}")
        
        # Group by sample
        samples = {}
        for member in members:
            # Extract sample key (everything before the extension)
            parts = member.name.split('.')
            if len(parts) >= 2:
                key = '.'.join(parts[:-1])
                ext = parts[-1]
                
                if key not in samples:
                    samples[key] = {}
                samples[key][ext] = member
        
        print(f"Total samples: {len(samples)}")
        
        # Show first few samples
        print(f"\nFirst {num_samples} samples:")
        for i, (key, files) in enumerate(list(samples.items())[:num_samples]):
            print(f"\n  Sample {i}: {key}")
            for ext, member in files.items():
                print(f"    .{ext}: {member.size} bytes")
                
                # Try to read and verify content
                if ext == 'jpg' or ext == 'png':
                    try:
                        f = tar.extractfile(member)
                        img_data = f.read()
                        img = Image.open(io.BytesIO(img_data))
                        print(f"      Image: {img.size}, mode: {img.mode}")
                    except Exception as e:
                        print(f"      ⚠️  Error reading image: {e}")
                
                elif ext == 'cls':
                    try:
                        f = tar.extractfile(member)
                        cls_data = f.read().decode('utf-8')
                        print(f"      Class: '{cls_data}'")
                    except Exception as e:
                        print(f"      ⚠️  Error reading class: {e}")
        
        return len(samples)


def test_webdataset_loading(url_pattern, num_batches=5):
    """Test loading data through WebDataset pipeline."""
    print(f"\n\nTesting WebDataset loading: {url_pattern}")
    print("="*60)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def decode_cls(data):
        return data.decode("utf-8")
    
    # Load class mapping
    import os
    wnids_path = "data/tiny-imagenet-200/wnids.txt"
    with open(wnids_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    def apply_transform(sample):
        image, label = sample
        label_id = class_to_idx.get(label, -1)
        if label_id == -1:
            print(f"⚠️  Unknown class: {label}")
        return transform(image), label_id
    
    # Create dataset
    try:
        dataset = (
            wds.WebDataset(url_pattern, shardshuffle=False)
            .decode(wds.handle_extension("cls", decode_cls), "pil")
            .to_tuple("jpg;png", "cls")
            .map(apply_transform)
        )
        
        # Create dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0)
        
        print("\nLoading batches...")
        total_samples = 0
        all_labels = []
        
        for i, (images, labels) in enumerate(loader):
            if i >= num_batches:
                break
            
            batch_size = images.shape[0]
            total_samples += batch_size
            all_labels.extend(labels.numpy())
            
            print(f"\nBatch {i}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Image stats: min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}")
            print(f"  Label range: min={labels.min()}, max={labels.max()}")
            print(f"  Sample labels: {labels[:5].tolist()}")
            
            # Check for issues
            if torch.isnan(images).any():
                print(f"  ⚠️  WARNING: NaN values in images!")
            if torch.isinf(images).any():
                print(f"  ⚠️  WARNING: Inf values in images!")
            if (labels < 0).any():
                print(f"  ⚠️  WARNING: Invalid labels (< 0)!")
            if (labels >= 200).any():
                print(f"  ⚠️  WARNING: Invalid labels (>= 200)!")
        
        print(f"\n✓ Successfully loaded {total_samples} samples")
        print(f"  Unique labels: {len(set(all_labels))}/200")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading WebDataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_datasets():
    """Compare WebDataset with raw data."""
    print("\n" + "="*60)
    print("WEBDATASET INTEGRITY CHECK")
    print("="*60)
    
    # Check training data
    print("\n" + "-"*60)
    print("TRAINING DATA")
    print("-"*60)
    
    train_tars = [
        "data/tiny-imagenet-wds/train/train-000000.tar",
        "data/tiny-imagenet-wds/train/train-000001.tar",
        "data/tiny-imagenet-wds/train/train-000002.tar",
    ]
    
    total_train_samples = 0
    for tar_path in train_tars:
        try:
            num_samples = inspect_tar_file(tar_path, num_samples=3)
            total_train_samples += num_samples
        except Exception as e:
            print(f"✗ Error inspecting {tar_path}: {e}")
    
    print(f"\nTotal training samples in WebDataset: {total_train_samples}")
    print(f"Expected: 100,000")
    
    # Check validation data
    print("\n" + "-"*60)
    print("VALIDATION DATA")
    print("-"*60)
    
    val_tars = [
        "data/tiny-imagenet-wds/val/val-000000.tar",
        "data/tiny-imagenet-wds/val/val-000001.tar",
    ]
    
    total_val_samples = 0
    for tar_path in val_tars:
        try:
            num_samples = inspect_tar_file(tar_path, num_samples=3)
            total_val_samples += num_samples
        except Exception as e:
            print(f"✗ Error inspecting {tar_path}: {e}")
    
    print(f"\nTotal validation samples in WebDataset: {total_val_samples}")
    print(f"Expected: 10,000")
    
    # Test loading through WebDataset pipeline
    print("\n" + "-"*60)
    print("TESTING WEBDATASET PIPELINE")
    print("-"*60)
    
    print("\n1. Testing Training Data Loading:")
    train_ok = test_webdataset_loading(
        "file:data/tiny-imagenet-wds/train/train-{000000..000002}.tar",
        num_batches=5
    )
    
    print("\n2. Testing Validation Data Loading:")
    val_ok = test_webdataset_loading(
        "file:data/tiny-imagenet-wds/val/val-{000000..000001}.tar",
        num_batches=5
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training samples: {total_train_samples} (expected: 100,000)")
    print(f"Validation samples: {total_val_samples} (expected: 10,000)")
    print(f"Training pipeline: {'✓ OK' if train_ok else '✗ FAILED'}")
    print(f"Validation pipeline: {'✓ OK' if val_ok else '✗ FAILED'}")
    
    if total_train_samples != 100000:
        print("\n⚠️  WARNING: Training sample count mismatch!")
    if total_val_samples != 10000:
        print("\n⚠️  WARNING: Validation sample count mismatch!")
    if not (train_ok and val_ok):
        print("\n⚠️  WARNING: WebDataset pipeline has issues!")


if __name__ == "__main__":
    compare_datasets()
