# models/resnet50/train.py
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.resnet50.models import ResNet50Model
from utils.imagedatasets import ImageDataset
from utils.load_config import load_config
from utils.preprocessing import label_to_idx

if __name__ == '__main__':
    # Set random seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)

    # Set batch size
    BATCH_SIZE = 16

    # Get the current date and time
    dt = datetime.now()

    # Format the datetime with custom separators
    f_dt = dt.strftime("%Y-%m-%d-%H-%M-%S")
    folder_name = f"run-{f_dt}"

    # Create a folder for artifacts
    os.makedirs(f"artifacts/{folder_name}", exist_ok=True)
    print(f"Folder name: {folder_name}")

    #create tensorboard writer
    writer=SummaryWriter(log_dir=f"artifacts/{folder_name}/tensorboard_logs")
    # Load configuration
    config = load_config("resnet50")
    print(config)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize early stopping parameters
    early_stopping_patience = 5 
    best_val_loss = float('inf')
    counter = 0  # Counter for how many epochs have occurred since an improvement in validation loss

    # Define transformations
    transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
   
    train_dataset = ImageDataset(csv_path=config['data']['train_csv'], transforms=transforms)

    # Assuming you have a train.csv file for each class, and you want to use a WeightedRandomSampler
    class_counts = {'No_DR': 1805, 'Mild': 370, 'Moderate': 999, 'Proliferate_DR': 295, 'Severe': 193}
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / (5 * count) for count in class_counts.values()]
    class_weights = [weight / max(class_weights) for weight in class_weights]
    # Print the class_weights to check
    print("Class Weights:", class_weights)

    # Convert train_dataset.labels to a PyTorch tensor
    # labels_tensor = torch.tensor(train_dataset.labels)

    # Create a list of weights corresponding to each sample
    # weights = [class_weights[label_to_idx(label)] for label in labels_tensor.cpu()]


    # # Create a list of weights corresponding to each sample
    weights =  torch.tensor([class_weights[label_to_idx(label)] for label in train_dataset.labels], dtype=torch.float32, device=device)

    # Convert the list of weights to a PyTorch tensor

    weights = torch.FloatTensor(weights.cpu())



    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # Create train_datasets and dataloaders
    train_dataset = ImageDataset(csv_path=config['data']['train_csv'], transforms=transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=config['model']['args']['batch_size'],  num_workers=config['model']['args']['num_workers'], sampler=sampler)

    # Create val_datasets and dataloaders
    val_dataset = ImageDataset(csv_path=config['data']['test_csv'], transforms=transforms)
    val_loader = DataLoader(val_dataset, batch_size=config['model']['args']['batch_size'],  num_workers=config['model']['args']['num_workers'],sampler=sampler)
    
    # Create model instance
    model = ResNet50Model(num_classes=config['model']['args']['num_classes']).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['model']['args']['learning_rate'])

    # Train the model
    num_epochs = config['model']['args']['num_epochs']
    save_interval = 1  # Save model checkpoint every `save_interval` epochs
    #unique_labels = set(train_dataset.data[i]['label'] for i in range(len(train_dataset)))
    unique_labels = set(sample[1] for sample in train_dataset)

    print("Unique Labels:", unique_labels)

    for epoch in range(num_epochs):
        
        train_running_loss=0
        val_running_loss=0
        train_running_accuracy=0
        val_running_accuracy=0

        model.train()
        #for i, (images, labels) in enumerate(train_loader):
        for i, (images, labels) in enumerate(train_loader):
            # Add debug prints or checks for indices
            print(f"Train Batch [{i + 1}/{len(train_loader)}], Index: {i * config['model']['args']['batch_size']}")


            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs=F.log_softmax(outputs,dim=1)
            loss=criterion(outputs,labels)
            train_running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            #         # Calculate weights and apply them to the loss
            # weights = [class_weights[label] for label in labels.numpy()]
            # weights = torch.tensor(weights, dtype=torch.float32, device=device)
            # weighted_loss = (loss * weights).mean()
            
            # calculate train_accuracy
            preds=torch.argmax(outputs,dim=1)
            accuracy=(preds==labels).float().mean()
            train_running_accuracy+=accuracy.item()
        
        #validation
        model.eval()
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)

        
            outputs = model(images)
            outputs=F.log_softmax(outputs,dim=1)

            loss=criterion(outputs,labels)
            val_running_loss+=loss.item()
            

            # calculate val_accuracy
            preds=torch.argmax(outputs,dim=1)
            accuracy=(preds==labels).float().mean()
            val_running_accuracy+=accuracy.item()

        avg_train_loss=train_running_loss/len(train_loader)
        avg_val_loss=val_running_loss/len(val_loader)
        



        avg_val_running_accuracy=val_running_accuracy/len(val_loader)
        avg_train_running_accuracy=train_running_accuracy/len(train_loader)

            # Log training loss to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar("Loss/val",avg_val_loss, epoch)
        writer.add_scalar("accuracy/train",avg_train_running_accuracy,epoch)
        writer.add_scalar("accuracy/val",avg_val_running_accuracy,epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break  # End training

        # Save model checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"models/resnet50/checkpoint_epoch_{epoch + 1}.pth")

    # Save the trained model
    torch.save(model.state_dict(), f"models/resnet50/{config['model']['name']}_model.pth")
