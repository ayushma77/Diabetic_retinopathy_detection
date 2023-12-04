import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms

# Import your custom dataset and other necessary modules
from utils.augumenting import CustomDataset

num_classes=5
batch_size=32
learning_rate=0.01
#Set Device and Define Transformations:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (you can modify these based on your requirements)
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other transformations as needed
])
#Load Augmented Data Using CustomDataset:
train_dataset = CustomDataset(root_dir="augmented_data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)



# Choose Model Architecture:
model = models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)  # Adjust based on your number of classes

# Move model to device
model = model.to(device)
#Define Loss Function and Optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#Train the Model:
num_epochs = 10  # Adjust as needed
# Create a SummaryWriter
#writer = SummaryWriter(log_dir='logs')
if __name__ == '__main__':
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    

    # Optionally, add validation/validation loss calculation here
#Save the Trained Model
torch.save(model.state_dict(), "resnet50_model.pth")  # Adjust the filename as needed

