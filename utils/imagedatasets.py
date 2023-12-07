import os
from os.path import join

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.io import read_as_csv
from utils.preprocessing import image_transforms, label_transforms, read_image


#create data set
class DiabeticDataset(Dataset):
    def __init__(self,csv_file):
        data_root=r"data"
        train_csv=r"data\train.csv"
        self.csv_file=os.path.join(data_root,train_csv)
    # load csv
        train_files, train_labels = read_as_csv(self.csv_file)
    # Apply transformations
        self.imgs = np.array(
            [image_transforms(file, label) for file, label in zip(train_files, train_labels)]
    )
        self.labels = np.array([label_transforms(lab) for lab in train_labels])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,label=0):
        
        return self.imgs[label],self.labels[label]
    
#1.prepare da
class ImageDataset(Dataset):
    def __init__(self,csv_path,transforms=None):
        images,labels=read_as_csv(csv_path)

        self.images=images
        self.labels=labels
        self.transforms=transforms
    def __str__(self):
        return f"<ImageDataset with {self.__len__()} samples>"
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        if index < 0 or index >= len(self.images):
            raise IndexError(f"Index {index} is out of range for the dataset.")

        image_name=self.images[index]
        label_name=self.labels[index]
        image_path=join("data",label_name,image_name)
        image = Image.open(image_path).convert('RGB')
        label=label_transforms(label_name)

        if self.transforms:
            image=self.transforms(image)
        return image,label
