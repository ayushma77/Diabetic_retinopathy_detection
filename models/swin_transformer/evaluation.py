import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.swin_transformer.model import SwinTransformerModel
from utils.augumenting import CustomDataset
from utils.load_config import config_load

# Load configuration
config = config_load("models/swin_transformer/configs/config_swin_transformer.yaml")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model instance
model = SwinTransformerModel(num_classes=config['num_classes']).to(device)
model.load_state_dict(torch.load(f"models/swin_transformer/{config['model']['name']}_model.pth"))
model.eval()

# Create dataset and dataloader for evaluation
eval_dataset = CustomDataset(root_dir=config['data_root'], transform=None)  # No need for transformation during evaluation
eval_loader = DataLoader(eval_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

# Define loss function
criterion = nn.CrossEntropyLoss()

# Evaluate the model
total_correct = 0
total_samples = 0
with torch.no_grad():
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print(f"Accuracy: {accuracy * 100:.2f}%")
