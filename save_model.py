import torch
from model import create_model

def save_model(model, filename='medical_image_model.pth'):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

# Example usage
save_model(model)
