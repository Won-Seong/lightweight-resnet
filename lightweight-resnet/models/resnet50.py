from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
        
    def forward(self, x):
        return self.model(x)