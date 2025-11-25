import sys
import os
import torch
import torchvision.transforms as T
from PIL import Image

# Add src to path
sys.path.append(os.getcwd())

from src.dataset import get_imagenet_dataset, CLASS_TO_IDX

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def inspect_url(url, name, train=True):
    print(f"\nInspecting {name} data from: {url}")
    
    loader = get_imagenet_dataset(url, batch_size=1, num_workers=0, train=train)
    
    try:
        for i, (data, target) in enumerate(loader):
            if i >= 1:
                break
            
            label_idx = target.item()
            class_name = [k for k, v in CLASS_TO_IDX.items() if v == label_idx]
            
            print(f"Sample {i}: Label Index: {label_idx}, Class Name: {class_name[0] if class_name else 'Unknown'}")
            print(f"Image Shape: {data.shape}")
            print(f"Image Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")
            
            # Save image
            img_tensor = denormalize(data[0])
            img = T.ToPILImage()(img_tensor.clamp(0, 1))
            save_path = f"test/{name}_sample_{i}.jpg"
            img.save(save_path)
            print(f"Saved sample to {save_path}")
            
    except Exception as e:
        print(f"Error during iteration: {e}")

def inspect():
    val_url = "https://storage.googleapis.com/caso-estudio-2/tiny-imagenet-wds/val/val-000000.tar"
    inspect_url(val_url, "val", train=False)
    
    train_url = "https://storage.googleapis.com/caso-estudio-2/tiny-imagenet-wds/train/train-{000000..000002}.tar"
    inspect_url(train_url, "train", train=True)

if __name__ == "__main__":
    inspect()
