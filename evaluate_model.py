import torch
from model import create_model
from data_preprocessing import MedicalImageDataset, DataLoader
from sklearn.metrics import accuracy_score

# Load the trained model
model = create_model(num_classes=5)  # Replace with your number of classes
model.load_state_dict(torch.load('medical_image_model.pth'))
model.eval()  # Set the model to evaluation mode

# Prepare the test set (assuming you have a separate test CSV)
test_dataset = MedicalImageDataset(csv_file='test_data.csv', root_dir='test_images/', transform=None)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate the model
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_dataloader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy:.4f}')
