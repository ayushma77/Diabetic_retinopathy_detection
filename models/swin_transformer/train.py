import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.swin_transformer.model import SwinTransformerModel
from utils.augumenting import CustomDataset
from utils.load_config import config_load

# Load configuration
config = config_load("models/swin_transformer/configs/config_swin_transformer.yaml")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
train_dataset = CustomDataset(root_dir=config['data_root'], transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

# Create model instance
model =  SwinTransformerModel(num_classes=config['num_classes']).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Create a SummaryWriter for TensorBoard
writer = SummaryWriter(log_dir='logs')

# Train the model
num_epochs = config['num_epochs']
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
    # ...

# Save the trained model
torch.save(model.state_dict(), f"models/swin_transformer/{config['model']['name']}_model.pth")
