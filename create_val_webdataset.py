"""
Convert Tiny ImageNet validation data to WebDataset format.
"""

import os
import tarfile
import shutil
from pathlib import Path
from PIL import Image
import io


def create_validation_webdataset(
    val_dir="data/tiny-imagenet-200/val",
    output_dir="data/tiny-imagenet-wds/val",
    samples_per_shard=1000
):
    """
    Convert Tiny ImageNet validation data to WebDataset format.
    
    Args:
        val_dir: Path to validation directory
        output_dir: Output directory for WebDataset shards
        samples_per_shard: Number of samples per tar file
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read validation annotations
    annotations_file = os.path.join(val_dir, "val_annotations.txt")
    images_dir = os.path.join(val_dir, "images")
    
    print(f"Reading annotations from: {annotations_file}")
    
    # Parse annotations: filename -> class_id
    annotations = {}
    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                filename = parts[0]
                class_id = parts[1]
                annotations[filename] = class_id
    
    print(f"Found {len(annotations)} validation images")
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.JPEG')])
    
    print(f"Processing {len(image_files)} images...")
    
    # Create shards
    shard_idx = 0
    sample_idx = 0
    current_tar = None
    
    for i, img_filename in enumerate(image_files):
        # Start new shard if needed
        if i % samples_per_shard == 0:
            if current_tar:
                current_tar.close()
                print(f"  Completed shard {shard_idx-1}")
            
            shard_path = os.path.join(output_dir, f"val-{shard_idx:06d}.tar")
            current_tar = tarfile.open(shard_path, 'w')
            print(f"Creating shard {shard_idx}: {shard_path}")
            shard_idx += 1
            sample_idx = 0
        
        # Get class label
        class_label = annotations.get(img_filename)
        if not class_label:
            print(f"Warning: No annotation for {img_filename}, skipping")
            continue
        
        # Read image
        img_path = os.path.join(images_dir, img_filename)
        
        # Create sample key (unique identifier)
        sample_key = f"{shard_idx-1:06d}_{sample_idx:06d}"
        
        # Add image to tar
        with open(img_path, 'rb') as img_file:
            img_data = img_file.read()
            img_info = tarfile.TarInfo(name=f"{sample_key}.jpg")
            img_info.size = len(img_data)
            current_tar.addfile(img_info, io.BytesIO(img_data))
        
        # Add class label to tar
        cls_data = class_label.encode('utf-8')
        cls_info = tarfile.TarInfo(name=f"{sample_key}.cls")
        cls_info.size = len(cls_data)
        current_tar.addfile(cls_info, io.BytesIO(cls_data))
        
        sample_idx += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images")
    
    # Close last shard
    if current_tar:
        current_tar.close()
        print(f"  Completed shard {shard_idx-1}")
    
    print(f"\n✓ Created {shard_idx} validation shards in {output_dir}")
    print(f"  Total samples: {len(image_files)}")
    print(f"  Samples per shard: ~{samples_per_shard}")
    
    # Print shard pattern for use in dataset loading
    if shard_idx == 1:
        pattern = f"file:{output_dir}/val-000000.tar"
    else:
        pattern = f"file:{output_dir}/val-{{000000..{shard_idx-1:06d}}}.tar"
    
    print(f"\nUse this pattern in your code:")
    print(f"  {pattern}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert validation data to WebDataset format')
    parser.add_argument('--val_dir', type=str, default='data/tiny-imagenet-200/val',
                        help='Path to validation directory')
    parser.add_argument('--output_dir', type=str, default='data/tiny-imagenet-wds/val',
                        help='Output directory for WebDataset shards')
    parser.add_argument('--samples_per_shard', type=int, default=1000,
                        help='Number of samples per shard')
    
    args = parser.parse_args()
    
    create_validation_webdataset(
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        samples_per_shard=args.samples_per_shard
    )
