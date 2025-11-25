import webdataset as wds
import torch
from torch.utils.data import DataLoader
import os


class TransformApplier:
    def __init__(self, train=True):
        self.transform = make_transform(train=train)
    
    def __call__(self, sample):
        image, label = sample
        # Hash the class string to an integer for simulation purposes
        # In production, use a proper mapping (wnids.txt)
        label_id = int(hash(label) % 200)
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

    dataset = (
        wds.WebDataset(url, shardshuffle=1000 if train else False)
        .shuffle(1000)
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
            T.RandomResizedCrop(224),
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
