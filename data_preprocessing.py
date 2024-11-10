import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

class MedicalImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.dataset = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        # Create a label mapping for your labels (assuming 'Finding Labels' are categorical)
        self.label_mapping = {label: idx for idx, label in enumerate(self.dataset['Finding Labels'].unique())}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the image file name from the CSV
        img_name = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])  # Assuming 'Image Index' is the first column
        
        # Check if the image file exists
        if not os.path.exists(img_name):
            print(f"Warning: Image {img_name} not found. Skipping this image.")
            return None  # Return None if the image doesn't exist, skipping the entry

        # Open the image
        image = Image.open(img_name).convert("RGB")
        
        # Get the label and map it to an index
        label = self.dataset.iloc[idx, 1]  # Assuming 'Finding Labels' is the second column
        label = self.label_mapping.get(label, -1)  # Use -1 if label is not found in the mapping

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label
