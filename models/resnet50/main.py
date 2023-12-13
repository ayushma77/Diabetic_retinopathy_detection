#1.prepare datasets
import os
from datetime import datetime
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import GaussianBlur

from models.resnet50.models import ResNet50Model
from utils.imagedatasets import ImageDataset
from utils.load_config import load_config

#training

if __name__=="__main__":
    best_val_accurcy=0
    SEED=42
    torch.manual_seed(SEED)
    config = load_config("resnet50")
    BATCH_SIZE=config['model']['args']['batch_size']
    num_workers=config['model']['args']['num_workers']
    #create folder
    #get the current date and time
    dt=datetime.now()
    #format the datime with custom separators
    f_dt=dt.strftime("%Y-%m-%d-%H-%M-%S")
    folder_name=f"run-{f_dt}"

    os.mkdir(f"artifacts/{folder_name}")
    print(f"folder name:{folder_name}")

    #create tensorboard writer
    writer=SummaryWriter(log_dir=f"artifacts/{folder_name}/tensorboard_logs")
    #prepare dataset
    train_csv_path=config['data']['train_csv']
    val_csv_path=config['data']['test_csv']
    # Assuming you have access to the number of samples in each class
    class_samples = [1805, 370, 999,295,193]  # Replace with actual numbers
    class_weights = 1.0 / torch.tensor(class_samples, dtype=torch.float32)
    # Create a WeightedRandomSampler

    # Define transformations
    transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        GaussianBlur(kernel_size=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset=ImageDataset(csv_path=train_csv_path,transforms=transforms)
    sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(train_dataset), replacement=True)

    train_data_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=False,
        sampler=sampler
        
    )
    val_dataset=ImageDataset(csv_path=val_csv_path,transforms=transforms)
    val_data_loader=DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=False,
        sampler=sampler
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet50Model(num_classes=config['model']['args']['num_classes']).to(device)
    #training
    LR=0.001
    EPOCHS=10
    
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    epochwise_train_loss=[]
    epochwise_val_loss=[]
    epochwise_val_accuracy=[]
    epochwise_train_accuracy=[]

    for epoch in range(EPOCHS):
        
        train_running_loss=0
        val_running_loss=0
        train_running_accuracy=0
        val_running_accuracy=0
    
        #traing
        model.train()
        for images,labels in train_data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            model_out=model(images)
            model_out=F.log_softmax(model_out,dim=1)

            loss=criterion(model_out,labels)
            train_running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            # calculate train_accuracy
            preds=torch.argmax(model_out,dim=1)
            accuracy=(preds==labels).float().mean()
            train_running_accuracy+=accuracy.item()
    
            
        #validation
        model.eval()
        

        for images,labels in val_data_loader:
            images, labels = images.to(device), labels.to(device)
            model_out=model(images)
            model_out=F.log_softmax(model_out,dim=1)

            loss=criterion(model_out,labels)
            val_running_loss+=loss.item()
            loss.backward()
            
        # calculate val_accuracy
            preds=torch.argmax(model_out,dim=1)
            accuracy=(preds==labels).float().mean()
            val_running_accuracy+=accuracy.item()
        avg_train_loss=train_running_loss/len(train_data_loader)
        avg_val_loss=val_running_loss/len(val_data_loader)
        avg_val_running_accuracy=val_running_accuracy/len(val_data_loader)
        avg_train_running_accuracy=train_running_accuracy/len(train_data_loader)

        if avg_val_running_accuracy>best_val_accurcy:
            best_val_accurcy=avg_val_running_accuracy
            torch.save(model.state_dict(),f"artifacts/{folder_name}/best_model.pth")

        #log to tensorboard
        writer.add_scalar("Loss/Train",avg_train_loss,epoch)
        writer.add_scalar("Loss/val",avg_val_loss,epoch)
        writer.add_scalar("accuracy/train",avg_train_running_accuracy,epoch)
        writer.add_scalar("accuracy/val",avg_val_running_accuracy,epoch)


        epochwise_train_loss.append(avg_train_loss)
        epochwise_val_loss.append(avg_val_loss)

        epochwise_val_accuracy.append(avg_val_running_accuracy)
        epochwise_train_accuracy.append(avg_train_running_accuracy)
        if len(epochwise_val_accuracy)>=6:
            avg_epochwise_val_accuracy_for_first_5epoch=np.mean(epochwise_val_accuracy[-5:])
            print(avg_epochwise_val_accuracy_for_first_5epoch)
            avg_epochwise_val_accuracy_for_last_5epoch=np.mean(epochwise_val_accuracy[-6:-1])
            print(avg_epochwise_val_accuracy_for_last_5epoch)
            diff=avg_epochwise_val_accuracy_for_first_5epoch-avg_epochwise_val_accuracy_for_last_5epoch
            print(f"diff:{diff:.3f}")
            if diff<0.001:
                break
        print(f"Epoch {epoch} Train_Loss:{avg_train_loss:.3f} \t Val Loss:{avg_val_loss:.3f}  \t Tain Accuracy:{avg_train_running_accuracy:.2f}  \t Val Accuracy:{avg_val_running_accuracy:.2f}")
        checkpoint_name=f"artifacts/{folder_name}/ckpt-{model.__class__.__name__}-val_acc-{avg_val_running_accuracy:.2f}-epoch-{epoch}.pth"
        checkpoint={'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'train_loss':avg_train_loss,
                    'val_loss':avg_val_loss,
                    'train_acc':avg_train_running_accuracy,
                    'val_acc':avg_val_running_accuracy
                }
        best_checkpoint_name=f"artifacts/{folder_name}/ckpt-{model.__class__.__name__}-best_val_acc-{best_val_accurcy:.2f}-epoch-{epoch}.pth"
        
        #save the model
        torch.save(checkpoint,best_checkpoint_name)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,9))
    x_axis_values=np.arange(0,EPOCHS,1)
    ax1.plot(epochwise_train_accuracy,label='train accuracy')
    ax1.plot(epochwise_val_accuracy,label='val accuracy')
    ax1.set_title("train vs val accuracy")
    ax1.legend()
    ax2.plot(epochwise_val_loss,label='val loss')
    ax2.plot(epochwise_train_loss,label='train loss')
    ax2.set_title("train vs val loss")
    ax2.legend()
    plt.title("Loss vs accuracy")
    plt.show()
