"""
Reorganize Tiny-ImageNet validation data to ImageFolder structure.

Original structure:
  val/
    images/
      val_0.JPEG
      val_1.JPEG
      ...
    val_annotations.txt

Target structure:
  val/
    n03444034/
      val_0.JPEG
      ...
    n04067472/
      val_1.JPEG
      ...
"""

import os
import shutil
from pathlib import Path


def reorganize_val_data(val_dir='data/tiny-imagenet-200/val'):
    """
    Reorganize validation data into ImageFolder structure.
    """
    
    val_dir = Path(val_dir)
    images_dir = val_dir / 'images'
    annotations_file = val_dir / 'val_annotations.txt'
    
    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist")
        return
    
    if not annotations_file.exists():
        print(f"Error: {annotations_file} does not exist")
        return
    
    print(f"Reading annotations from {annotations_file}")
    
    # Read annotations
    annotations = {}
    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                filename = parts[0]
                class_id = parts[1]
                annotations[filename] = class_id
    
    print(f"Found {len(annotations)} annotations")
    
    # Check if already reorganized
    class_dirs = [d for d in val_dir.iterdir() if d.is_dir() and d.name.startswith('n')]
    if class_dirs:
        print(f"\nValidation data appears to already be reorganized ({len(class_dirs)} class directories found)")
        response = input("Do you want to reorganize again? This will reset the structure. (y/n): ")
        if response.lower() != 'y':
            print("Skipping reorganization")
            return
        
        # Remove existing class directories
        print("Removing existing class directories...")
        for class_dir in class_dirs:
            shutil.rmtree(class_dir)
        
        # Move images back to images/ if needed
        if not images_dir.exists():
            print("Recreating images directory...")
            images_dir.mkdir()
    
    # Create class directories and move images
    print("\nReorganizing images by class...")
    
    moved_count = 0
    for filename, class_id in annotations.items():
        # Create class directory if it doesn't exist
        class_dir = val_dir / class_id
        class_dir.mkdir(exist_ok=True)
        
        # Move image to class directory
        src = images_dir / filename
        dst = class_dir / filename
        
        if src.exists():
            shutil.move(str(src), str(dst))
            moved_count += 1
            
            if moved_count % 1000 == 0:
                print(f"  Moved {moved_count}/{len(annotations)} images")
        else:
            print(f"Warning: {src} does not exist")
    
    print(f"\n✓ Moved {moved_count} images to class directories")
    
    # Remove empty images directory
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()
        print(f"✓ Removed empty {images_dir}")
    
    # Count classes
    class_dirs = [d for d in val_dir.iterdir() if d.is_dir() and d.name.startswith('n')]
    print(f"✓ Created {len(class_dirs)} class directories")
    
    print("\n" + "="*60)
    print("Validation data reorganization complete!")
    print("="*60)
    print(f"Structure is now compatible with ImageFolder")
    print(f"You can now use: datasets.ImageFolder('{val_dir}')")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reorganize Tiny-ImageNet validation data')
    parser.add_argument('--val_dir', type=str, default='data/tiny-imagenet-200/val',
                        help='Path to validation directory')
    
    args = parser.parse_args()
    
    reorganize_val_data(args.val_dir)
