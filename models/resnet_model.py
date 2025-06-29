from torchvision import models
import torch.nn as nn

def get_resnet18(num_classes=2):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model