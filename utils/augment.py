import csv
import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

# Define your data directory
data_root = "data"
label_to_idx_map = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Proliferate_DR': 3, 'Severe': 4}
idx_to_label_map = {idx: label for label, idx in label_to_idx_map.items()}


# # Define transformations for augmentation and normalization
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(256),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(20),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def read_as_csv(csv_file):
    image_path= []
    labels= []
    with open(csv_file , 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            image_path.append(row[0])
            labels.append(row[1])
    return image_path, labels

class CustomDataset(Dataset):
    def __init__(self, csv_file, data_root="data",transform=None):
        self.data_root = data_root
        self.csv_file = csv_file
        self.transform=transform
        self.imgs, self.labels = self._load_data()
    



    def label_to_idx(self,label):
        return label_to_idx_map[label]
    def idx_to_label(self,idx):
        """Convert index to label."""
        try:
            return idx_to_label_map[idx]
        except KeyError:
            raise KeyError(f"Label not found. Try one of these: {idx_to_label_map.keys()}")



    def _read_image(self, file_path, mode, resize=(256, 256), grayscale=False):
        print("Trying to open:", file_path)

        # Implement your image reading logic here
        image = Image.open(file_path)

        if grayscale:
            image = image.convert("L")

        image = Image.open(file_path)
        # print(image.size)
        height,width=image.size
        if height==width:
            pass
        else:
            
            if mode=='zoom':
                # left=0
                # right=0
                # upper=0
                # lower=0
                if height<width:
                    diff=width-height
                    left=diff//2
                    right=width-diff//2
                    upper=0
                    lower=height
                elif width<height:
                    diff=height-width
                    left=0
                    right=width
                    upper=diff//2
                    lower=height-upper


                new_image=image.crop((left,upper,right,lower))
            elif mode=='padding':
            
                image=ImageOps.pad(image, size=(256,256), centering=(0.5, 0.5))
                
        new_image=image.resize(resize)
        img_array = np.asarray(new_image)
        return img_array
    
    # def _image_transforms(self, file_name, label):
    #     file_path = os.path.join(self.data_root, label, file_name)
    #     array = self._read_image(file_path, "padding", grayscale=True)
    #     flatten_image = array.flatten()
    #     return flatten_image
    def _image_transforms(self, file_name, label):
        file_path = os.path.join(self.data_root, label, file_name)
        file_path=os.path.abspath(file_path)
        try:
            

        # Apply transformations
            transform = transforms.Compose([
            # Add your desired transformations here
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
            image = Image.open(file_path).convert('RGB')
            image = transform(image)

            return image
        except FileNotFoundError:
            # Handle missing file (e.g., print a message, skip the image, or remove the entry)
            print(f"File not found: {file_path}")
            return None
            
    def _label_transforms(self, label) -> int:
        try:
            return label_to_idx_map[label]
        except KeyError:
            print(f"Label not found: {label}")
            return -1  # Or any other value to indicate an error


    def _load_data(self):
        train_files, train_labels = read_as_csv(self.csv_file)
        imgs = [self._image_transforms(file, label) for file, label in zip(train_files, train_labels)]

        labels = [self._label_transforms(lab) for lab in train_labels]
        labels = torch.tensor(labels, dtype=torch.long)

        return imgs, labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

# Example usage
csv_file_path = "data/train.csv"
dataset = CustomDataset(csv_file_path)

# Assuming you have a train.csv file for each class, and you want to use a WeightedRandomSampler
class_counts = {'No_DR': 1805, 'Mild': 370, 'Moderate': 999, 'Proliferate_DR': 295, 'Severe': 193}
total_samples = sum(class_counts.values())
class_weights = [total_samples / (5 * count) for count in class_counts.values()]
class_weights = [weight / max(class_weights) for weight in class_weights]
# Print the class_weights to check
print("Class Weights:", class_weights)
# Create a list of weights corresponding to each sample


# Create a list of weights corresponding to each sample
weights = [class_weights[label] for label in dataset.labels]

# Convert the list of weights to a PyTorch tensor
weights = torch.DoubleTensor(weights)

# Create a WeightedRandomSampler
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

# Create DataLoader with the WeightedRandomSampler
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# Save the augmented images to the new directory
augmented_data_dir = "data/augmented_data"
os.makedirs(augmented_data_dir, exist_ok=True)

# Create lists to store the paths of the augmented images and their labels
augmented_images = []
augmented_labels = []

for idx, (images, labels) in enumerate(train_loader):
    try:
        for image, label in zip(images, labels):
            class_dir = os.path.join(augmented_data_dir, idx_to_label_map[int(label)])
            os.makedirs(class_dir, exist_ok=True)

            # Generate a unique filename for each augmented image
            image_filename = f"augmented_image_{idx}_{label}.png"
            image_path = os.path.join(class_dir, image_filename)
            print("Saving:", image_path) 
            torchvision.utils.save_image(image, image_path)

            # Append the path and label to the lists
            augmented_images.append(image_path)
            augmented_labels.append(int(label))

    except IndexError as e:
        print(f"Error: {e}")
        print(f"Index: {idx}, Images Shape: {images.shape}, Labels Shape: {labels.shape}")

# Create a DataFrame for augmented data
augmented_df = pd.DataFrame({"file": augmented_images, "label": augmented_labels})


# Load the original train CSV
train_csv_path = os.path.join(data_root, "train.csv")
train_df = pd.read_csv(train_csv_path)
# Convert the paths in the original train_df to a consistent format
train_df['file'] = train_df['file'].apply(lambda x: os.path.relpath(os.path.join(data_root, x), data_root))
# Convert the paths in the augmented_df to a consistent format
augmented_df['file'] = augmented_df['file'].apply(lambda x: os.path.relpath(x, data_root))

# Concatenate the original and augmented DataFrames
merged_df = pd.concat([train_df, augmented_df], ignore_index=True)


# Save the merged DataFrame to a new CSV file
merged_csv_path = os.path.join(data_root, "merged_train.csv")
merged_df.to_csv(merged_csv_path, index=False)
