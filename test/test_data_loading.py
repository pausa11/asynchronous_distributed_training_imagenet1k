"""
Test script to validate Tiny ImageNet data loading.
This will help identify if we have a "garbage in, garbage out" problem.
"""

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Path to Tiny ImageNet
DATA_DIR = "/Users/danieltorosoto/universidad/arq-cliente-servidor/asynchronous_distributed_training_imagenet1k/data/tiny-imagenet-200"

def load_class_names():
    """Load the class names from wnids.txt"""
    wnids_path = os.path.join(DATA_DIR, "wnids.txt")
    with open(wnids_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def create_class_to_idx(class_names):
    """Create a mapping from class name to index"""
    return {class_name: idx for idx, class_name in enumerate(class_names)}

def test_train_data():
    """Test the training data loading"""
    print("=" * 80)
    print("TESTING TRAINING DATA")
    print("=" * 80)
    
    # Simple transform for testing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # Load training data using ImageFolder
    train_dir = os.path.join(DATA_DIR, "train")
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  - Number of samples: {len(train_dataset)}")
    print(f"  - Number of classes: {len(train_dataset.classes)}")
    print(f"  - First 10 classes: {train_dataset.classes[:10]}")
    
    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\n✓ Batch loaded successfully!")
    print(f"  - Batch shape: {images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Labels in batch: {labels[:10].tolist()}")
    print(f"  - Unique labels in batch: {torch.unique(labels).tolist()}")
    print(f"  - Min/Max pixel values: {images.min():.3f} / {images.max():.3f}")
    
    # Check if labels are in valid range
    assert labels.min() >= 0 and labels.max() < 200, "Labels out of range!"
    print(f"\n✓ Labels are in valid range [0, 199]")
    
    # Display a few images
    print("\n✓ Displaying first 4 images from batch...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            img = images[idx].permute(1, 2, 0).numpy()
            # Denormalize if needed
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            class_name = train_dataset.classes[labels[idx]]
            ax.set_title(f"Class: {class_name}\nLabel: {labels[idx]}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('train_samples.png')
    print(f"  - Saved visualization to train_samples.png")
    
    return train_dataset, train_loader

def test_validation_data():
    """Test the validation data loading"""
    print("\n" + "=" * 80)
    print("TESTING VALIDATION DATA")
    print("=" * 80)
    
    # Simple transform for testing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # For Tiny ImageNet validation, we need to parse val_annotations.txt
    val_dir = os.path.join(DATA_DIR, "val")
    val_annotations_path = os.path.join(val_dir, "val_annotations.txt")
    
    # Load class names
    class_names = load_class_names()
    class_to_idx = create_class_to_idx(class_names)
    
    # Parse validation annotations
    val_data = []
    with open(val_annotations_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            img_name = parts[0]
            class_name = parts[1]
            img_path = os.path.join(val_dir, "images", img_name)
            label = class_to_idx[class_name]
            val_data.append((img_path, label))
    
    print(f"\n✓ Validation annotations loaded!")
    print(f"  - Number of validation samples: {len(val_data)}")
    print(f"  - First 5 samples:")
    for i in range(5):
        img_path, label = val_data[i]
        print(f"    {i+1}. {os.path.basename(img_path)} -> Label {label} ({class_names[label]})")
    
    # Load a few images manually
    print("\n✓ Loading and displaying first 4 validation images...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        if idx < 4:
            img_path, label = val_data[idx]
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            
            img_display = img_tensor.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
            ax.imshow(img_display)
            ax.set_title(f"Class: {class_names[label]}\nLabel: {label}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('val_samples.png')
    print(f"  - Saved visualization to val_samples.png")
    
    return val_data

def test_model_forward_pass():
    """Test a simple forward pass through a model"""
    print("\n" + "=" * 80)
    print("TESTING MODEL FORWARD PASS")
    print("=" * 80)
    
    # Create a simple ResNet18 model
    model = torchvision.models.resnet18(num_classes=200)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(4, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n✓ Model forward pass successful!")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Check predictions
    predictions = output.argmax(dim=1)
    print(f"  - Predictions: {predictions.tolist()}")
    print(f"  - All predictions in valid range: {(predictions >= 0).all() and (predictions < 200).all()}")
    
    return model

def main():
    print("\n" + "=" * 80)
    print("TINY IMAGENET DATA VALIDATION TEST")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Directory exists: {os.path.exists(DATA_DIR)}")
    
    try:
        # Test training data
        train_dataset, train_loader = test_train_data()
        
        # Test validation data
        val_data = test_validation_data()
        
        # Test model
        model = test_model_forward_pass()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nConclusion:")
        print("  - Data is loading correctly")
        print("  - Labels are in valid range [0, 199]")
        print("  - Images are properly formatted")
        print("  - Model can process the data")
        print("\nIf your training accuracy is 0%, the issue is likely:")
        print("  1. The WebDataset format you're using doesn't match the data")
        print("  2. Label mapping in dataset.py is incorrect (using hash instead of proper mapping)")
        print("  3. Training loop has bugs")
        print("\nCheck the generated images: train_samples.png and val_samples.png")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
