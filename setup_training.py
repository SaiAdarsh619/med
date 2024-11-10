import torch
import torch.optim as optim
import torch.nn as nn

def setup_training(model):
    # Use CrossEntropyLoss for single-label classification
    criterion = nn.CrossEntropyLoss()

    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return criterion, optimizer
