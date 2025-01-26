from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

def get_resnet50(out_features : int = None):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    if out_features is not None:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_features)
        print("Out features = " + str(out_features))
    return model