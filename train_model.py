import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from data_preprocessing import MedicalImageDataset  # Import your dataset class

def collate_fn(batch):
    # Filter out None values in the batch
    batch = [item for item in batch if item is not None]
    images, labels = zip(*batch)  # Unzip the images and labels
    return torch.stack(images, 0), torch.tensor(labels)  # Convert to tensors


# Step 1: Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (ResNet expected size)
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

# Step 2: Load the dataset and create DataLoader
dataset = MedicalImageDataset(csv_file=r'C:\med\labes.csv\Filtered_Data_Entry.csv', root_dir='images', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Step 3: Load pre-trained ResNet model and modify the final layer
model = models.resnet18(pretrained=True)

# Modify the final layer for the number of classes in your dataset
num_classes = len(dataset.label_mapping)  # Assuming your dataset has a label mapping
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Step 4: Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Step 6: Train the model
num_epochs = 10  # You can adjust the number of epochs

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        if images is None:
            continue
        images, labels = images.to(device), labels.to(device)  # Move data to GPU if available

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss

        loss.backward()  # Backward pass
        optimizer.step()  # Optimization step

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

# Step 7: Save the trained model
torch.save(model.state_dict(), 'medical_image_model.pth')
print("Model saved as medical_image_model.pth")
