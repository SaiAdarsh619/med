import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes):
    # Load a pre-trained ResNet model (ResNet-18)
    model = models.resnet18(pretrained=True)
    
    # Modify the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
