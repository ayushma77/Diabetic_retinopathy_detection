import os

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.data.iloc[idx, 1])  # Assuming labels are integers

        if self.transform:
            image = self.transform(image)

        return image, label

# Augmentation transforms
augmentation_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
])

# Define train_dataset and train_loader
train_dataset = CustomDataset(csv_file=r'data/train.csv', root_dir=r'data', transform=augmentation_transform)



# Create a mapping from class names to integers
class_name_to_int = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Proliferate_DR': 3, 'Severe': 4}

# Convert the string labels to integers using the mapping
train_dataset.data['label'] = train_dataset.data['label'].map(class_name_to_int)






# Assuming you have a train.csv file for each class, and you want to use a WeightedRandomSampler
class_counts = {'No_DR': 1805, 'Mild': 370, 'Moderate': 999, 'Proliferate_DR': 295, 'Severe': 193}
total_samples = sum(class_counts.values())
class_weights = [total_samples / (5 * count) for count in class_counts.values()]
class_weights = [weight / max(class_weights) for weight in class_weights]

# Create a list of weights corresponding to each sample
weights = [class_weights[label] for label in train_dataset.data['label']]

# Convert the list of weights to a PyTorch tensor
weights = torch.DoubleTensor(weights)

# Create a WeightedRandomSampler
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

import matplotlib.pyplot as plt
import numpy as np


# Function to display a batch of images
def show_images(images, labels):
    batch_size = images.size(0)
    grid = torchvision.utils.make_grid(images, nrow=int(np.sqrt(batch_size)), padding=5)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(f'Labels: {labels}')
    plt.axis('off')
    plt.show()

# Assuming you have a train_loader
for batch_idx, (data, target) in enumerate(train_loader):
    # Display a few examples from the batch
    show_images(data, target)

    # Break the loop to display only the first batch
    break
