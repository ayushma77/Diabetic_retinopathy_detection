import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from models.resnet50.models import ResNet50Model
from utils.augumenting import CustomDataset
from utils.load_config import config_load

# Load configuration
config = config_load("models/resnet/configs/config_resnet.yaml")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model instance
model = ResNet50Model(num_classes=config['num_classes']).to(device)
model.load_state_dict(torch.load(f"models/resnet/{config['model']['name']}_model.pth"))
model.eval()

# Create dataset and dataloader for evaluation
eval_dataset = CustomDataset(root_dir=config['data_root'], transform=None)  # No need for transformation during evaluation
eval_loader = DataLoader(eval_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

# Lists to store true labels and predicted labels
all_true_labels = []
all_predicted_labels = []

# Evaluate the model and collect predictions
with torch.no_grad():
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_true_labels.extend(labels.cpu().numpy())
        all_predicted_labels.extend(predicted.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(all_true_labels, all_predicted_labels)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config['classes'], yticklabels=config['classes'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
report = classification_report(all_true_labels, all_predicted_labels, target_names=config['classes'])
print("Classification Report:\n", report)


