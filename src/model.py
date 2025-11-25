import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=1000, pretrained=False):
    """
    Returns a ResNet50 model.
    Args:
        num_classes (int): Number of output classes (1000 for ImageNet, 200 for Tiny-ImageNet).
        pretrained (bool): Whether to use pretrained weights.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    
    # Replace the final fully connected layer if num_classes is different
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    return model
