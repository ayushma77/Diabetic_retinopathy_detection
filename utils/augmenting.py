
import os

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

# Define your data directory
data_root = "data"
label_to_idx_map = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Proliferate_DR': 3, 'Severe': 4}
idx_to_label_map = {idx: label for label, idx in label_to_idx_map.items()}

def label_to_idx(label):
    return label_to_idx_map[label]
def idx_to_label(idx):
    """Convert index to label."""
    try:
        return idx_to_label_map[idx]
    except KeyError:
        raise KeyError(f"Label not found. Try one of these: {idx_to_label_map.keys()}")


# Define transformations for augmentation and normalization
transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, csv_path, root_dir, label_to_idx_func, transform=None):
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.label_to_idx_func = label_to_idx_func
        self.transform = transform
        self.data = self._load_data()

    def label_to_idx(self, label):
        return self.label_to_idx_func(label)

    def _load_data(self):
        data = []
        train_df = pd.read_csv(self.csv_path)
        for index, row in train_df.iterrows():
            img_path = row['file']  # Use row['file'] directly
            label = self.label_to_idx(row['label'])
            data.append({'image_path': img_path, 'label': label})
        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, str(self.data[idx]['label']), self.data[idx]['image_path'])

        label = self.data[idx]['label']

        print(f"Root Directory: {self.root_dir}")
        print(f"Row File: {self.data[idx]['image_path']}")
        print(f"Image Path: {img_path}")
        

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Index: {idx}, Image Path: {img_path}")
            raise e

        if self.transform:
            image = self.transform(image)

        return image, label






# Create an instance of the custom dataset
train_csv_path = os.path.join(data_root, "train.csv")
train_dataset = CustomDataset(csv_path=train_csv_path, root_dir=data_root, label_to_idx_func=label_to_idx, transform=transform)
# Calculate class weights
class_counts = {'No_DR': 1805, 'Mild': 370, 'Moderate': 999, 'Proliferate_DR': 295, 'Severe': 193}
total_samples = len(train_dataset)
class_weights = [total_samples / ((len([data for data in train_dataset.data if data['label'] == label]) * class_counts[label]) + 1) for label in class_counts]


# Create a sampler with weights for training data
weights = [class_weights[train_dataset.data[i]['label']] for i in range(len(train_dataset))]
sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)

# Create data loader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

# Define a new directory for augmented data
augmented_data_dir = "data/augmented_data"

# Save the augmented images to the new directory
for idx, (images, labels) in enumerate(train_loader):
    try:
        for image, label in zip(images, labels):
            class_dir = os.path.join(augmented_data_dir, idx_to_label(label))
            os.makedirs(class_dir, exist_ok=True)

            image_path = os.path.join(class_dir, f"augmented_image_{idx}.png")
            torchvision.utils.save_image(image, image_path)

    except IndexError as e:
        print(f"Error: {e}")
        print(f"Index: {idx}, Images Shape: {images.shape}, Labels Shape: {labels.shape}")

# Get the paths of the augmented images
augmented_images = [os.path.join(root, file) for root, dirs, files in os.walk(augmented_data_dir) for file in files if file.endswith('.png')]

# Create a DataFrame for augmented data
augmented_df = pd.DataFrame({"file": augmented_images, "label": [label_to_idx(os.path.basename(os.path.dirname(img))) for img in augmented_images]})

# Load the original train CSV
train_csv_path = os.path.join(data_root, "train.csv")
train_df = pd.read_csv(train_csv_path)

# Concatenate the original and augmented DataFrames
merged_df = pd.concat([train_df, augmented_df], ignore_index=True)


# Save the merged DataFrame to a new CSV file
merged_csv_path = os.path.join(data_root, "merged_train.csv")
merged_df.to_csv(merged_csv_path, index=False)
