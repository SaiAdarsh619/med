import torch
import cv2
from model import create_model
from torchvision import transforms
import matplotlib.pyplot as plt

# Load the trained model
model = create_model(num_classes=5)
model.load_state_dict(torch.load('medical_image_model.pth'))
model.eval()

# Prepare the image for inference
image = cv2.imread('00000003_004.png.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess the image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_tensor = preprocess(image_rgb).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)

# Annotate image with predicted class
label = predicted.item()  # Get predicted label

# Convert back to BGR for OpenCV to display
image_annotated = image.copy()
cv2.putText(image_annotated, f'Predicted: {label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the annotated image
cv2.imshow('Annotated Image', image_annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
