import webdataset as wds
import torch
from torch.utils.data import DataLoader
import os


def load_class_mapping():
    """
    Load the class-to-index mapping from wnids.txt.
    This ensures consistent label mapping across training and validation.
    """
    # Path to wnids.txt (relative to project root)
    wnids_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "tiny-imagenet-200", "wnids.txt"
    )
    
    if not os.path.exists(wnids_path):
        raise FileNotFoundError(f"wnids.txt not found at {wnids_path}")
    
    with open(wnids_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Create mapping from class name to index
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    return class_to_idx


# Load the class mapping once at module level
CLASS_TO_IDX = load_class_mapping()


class TransformApplier:
    def __init__(self, train=True):
        self.transform = make_transform(train=train)
    
    def __call__(self, sample):
        image, label = sample
        # Use proper class-to-index mapping
        label_id = CLASS_TO_IDX.get(label, -1)
        
        if label_id == -1:
            raise ValueError(f"Unknown class label: {label}")
        
        return self.transform(image), label_id

def decode_cls(data):
    return data.decode("utf-8")

def get_imagenet_dataset(url, batch_size=64, num_workers=4, train=True):
    """
    Creates a WebDataset loader for ImageNet/Tiny-ImageNet from a GCP bucket URL.
    
    Args:
        url (str): The URL pattern for the dataset (e.g., "gs://my-bucket/imagenet-train-{0000..0146}.tar").
                   Note: For GCS, you might need to use pipe:gsutil cat or similar if not public, 
                   or use http URLs if public. WebDataset supports various schemes.
                   If using local files or mounted bucket, use file paths.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        train (bool): Whether to load training or validation set (affects transforms).
    
    Returns:
        DataLoader: PyTorch DataLoader.
    """
    
    # Apply transforms
    applier = TransformApplier(train=train)

    # Build pipeline - shuffle only for training
    dataset = wds.WebDataset(url, shardshuffle=1000 if train else False)
    
    # Apply shuffle only during training, not validation
    if train:
        dataset = dataset.shuffle(1000)
    
    dataset = (
        dataset
        .decode(wds.handle_extension("cls", decode_cls), "pil")
        .to_tuple("jpg;png", "cls")
        .map(applier)
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    return loader

def make_transform(train=True):
    import torchvision.transforms as T
    
    if train:
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_imagenet_folder_dataset(data_dir, batch_size=64, num_workers=4, train=True):
    """
    Creates a DataLoader for ImageNet/Tiny-ImageNet from uncompressed directory structure.
    This matches the approach used in train_simple.py (proven to work).
    
    Args:
        data_dir (str): Path to the dataset directory (local or gcsfuse-mounted).
                       Should contain 'train/' and 'val/' subdirectories.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        train (bool): Whether to load training or validation set.
    
    Returns:
        DataLoader: PyTorch DataLoader.
    """
    from torchvision import datasets
    import os
    
    # Get transforms (same as WebDataset version)
    transform = make_transform(train=train)
    
    # Determine subdirectory
    subdir = 'train' if train else 'val'
    dataset_path = os.path.join(data_dir, subdir)
    
    # Use ImageFolder for easy loading
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,  # Shuffle only for training
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader
