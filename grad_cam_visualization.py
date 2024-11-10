import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from model import create_model
from torchvision import transforms
from data_preprocessing import MedicalImageDataset

# Load model
model = create_model(num_classes=5)  # Adjust for your dataset
model.load_state_dict(torch.load('medical_image_model.pth'))
model.eval()

# Choose a test image for Grad-CAM
image = cv2.imread('test_images/some_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess the image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_tensor = preprocess(image).unsqueeze(0)

# Grad-CAM Hook
def save_grad(grad):
    global gradients
    gradients = grad

def generate_gradcam(model, image_tensor):
    # Hook the last convolutional layer
    last_conv_layer = model.layer4[1].conv2
    last_conv_layer.register_hook(save_grad)

    # Forward pass
    output = model(image_tensor)
    class_idx = output.argmax(dim=1).item()

    # Backward pass to get gradients
    output[:, class_idx].backward()

    # Grad-CAM calculation
    grad = gradients[0].cpu().data.numpy()
    weight = np.mean(grad, axis=(1, 2))  # Global average pooling
    activation = last_conv_layer(image_tensor).cpu().data.numpy()[0]

    # Weighted sum of the activations
    cam = np.zeros_like(activation[0], dtype=np.float32)
    for i in range(activation.shape[0]):
        cam += weight[i] * activation[i, :, :]

    # Apply ReLU
    cam = np.maximum(cam, 0)

    # Normalize the heatmap
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

    return cam

# Generate Grad-CAM
cam = generate_gradcam(model, image_tensor)

# Visualize the Grad-CAM heatmap
plt.imshow(image)
plt.imshow(cam, cmap='jet', alpha=0.5)  # Overlay heatmap
plt.show()
