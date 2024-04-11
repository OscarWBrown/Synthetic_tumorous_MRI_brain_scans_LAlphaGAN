import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
import argparse
import seaborn as sns


# Assuming CUDA_VISIBLE_DEVICES is managed externally for PyTorch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
#device = torch.device("cpu")

class CustomDataset(Dataset):
    def __init__(self, directory, target_size=(64, 64), class_mode='binary', transform = None):
        self.directory = directory
        self.target_size = target_size
        self.class_mode = class_mode
        self.filenames, self.labels = self._load_filenames_and_labels()
        self.transform = transform

    def _load_filenames_and_labels(self):
        filenames = []
        labels = []
        # Modify according to your dataset structure
        class_dirs = {'notumor': 0, 'meningioma': 1, 'pituitary': 1, 'augmented_no_tumor': 0, 'glioma': 1}  
        for class_name, label in class_dirs.items():
            class_dir = os.path.join(self.directory, class_name)
            for fname in os.listdir(class_dir):
                filenames.append(os.path.join(class_dir, fname))
                labels.append(label)
        return filenames, np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.target_size)
        # img_tensor = transforms.ToTensor()(img)
        # img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
        img_tensor = self.transform(img)
        label = self.labels[idx]
        return img_tensor, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.bn3(self.fc1(self.dropout(x))))
        x = torch.sigmoid(self.fc2(self.dropout(x)))
        return x

def calculate_accuracy(outputs, labels):
    preds = outputs.round()
    return torch.sum(preds == labels.unsqueeze(1)).item() / len(labels)

# def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         running_accuracy = 0.0

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs.squeeze(), labels)
#             accuracy = calculate_accuracy(outputs, labels)
#             running_accuracy += accuracy
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_accuracy = running_accuracy / len(train_loader)

#         model.eval()
#         val_running_loss = 0.0
#         val_running_accuracy = 0.0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs.squeeze(), labels)
#                 accuracy = calculate_accuracy(outputs, labels)
#                 val_running_accuracy += accuracy
#                 val_running_loss += loss.item() * inputs.size(0)
#         val_epoch_loss = val_running_loss / len(val_loader.dataset)
#         val_epoch_accuracy = val_running_accuracy / len(val_loader)

#         print(f'Epoch {epoch+1}/{num_epochs}, '
#               f'Training Loss: {epoch_loss:.4f}, '
#               f'Training Accuracy: {epoch_accuracy:.4f}, '
#               f'Validation Loss: {val_epoch_loss:.4f}, '
#               f'Validation Accuracy: {val_epoch_accuracy:.4f}')

import matplotlib.pyplot as plt

def calculate_metrics(outputs, labels):
    # Convert outputs to predictions: output > 0.5 => 1, else => 0
    preds = outputs.round().cpu().numpy()
    labels = labels.cpu().numpy()
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = fbeta_score(labels, preds, beta=1)  # F1 score
    f2 = fbeta_score(labels, preds, beta=2)  # F2 score emphasizes recall more than precision
    return precision, recall, f1, f2

def save_confusion_matrix(conf_matrix, class_names, file_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    plt.savefig(f'{file_path}/confusion_matrix.png')
    plt.close(fig) 


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    # Initialize lists to store metrics per epoch
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'precision': [],
        'recall': [],
        'F1': [],
        'F2': []
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        #print(2)

        for inputs, labels in train_loader:
            #print(1)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            accuracy = calculate_accuracy(outputs, labels)
            running_accuracy += accuracy
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = running_accuracy / len(train_loader)

        model.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                accuracy = calculate_accuracy(outputs, labels)
                val_running_accuracy += accuracy
                val_running_loss += loss.item() * inputs.size(0)
                all_preds.extend(outputs.squeeze().tolist()) 
                all_labels.extend(labels.tolist()) 
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_accuracy = val_running_accuracy / len(val_loader)
        all_preds = torch.tensor(all_preds).to(device)
        all_labels = torch.tensor(all_labels).to(device)
        precision, recall, F1, F2 = calculate_metrics(all_preds, all_labels)

        # Append metrics to the history dict
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['val_accuracy'].append(val_epoch_accuracy)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['F1'].append(F1)
        history['F2'].append(F2)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}, F1: {F1:.4f}, F2: {F2:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

    # After the final epoch, compute the confusion matrix
    binary_preds = (all_preds >= 0.5).long()
    
    conf_matrix = confusion_matrix(all_labels.cpu().numpy(), binary_preds.cpu().numpy())
    
    
    return history, conf_matrix

def plot_and_save_metrics(history, dir, name):
    import matplotlib.pyplot as plt

def plot_and_save_metrics(history, dir, name):
    epochs = range(1, len(history['train_loss']) + 1)  # Assuming epoch count is the same for all metrics

    plt.figure(figsize=(20, 10))

    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for i in epochs:
        if i % 10 == 0:  # Annotate every 25 epochs
            plt.text(i, history['train_loss'][i-1], f'{history["train_loss"][i-1]:.2f}', ha='center', va='bottom')
            plt.text(i, history['val_loss'][i-1], f'{history["val_loss"][i-1]:.2f}', ha='center', va='top')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for i in epochs:
        if i % 10 == 0:  # Annotate every 25 epochs
            plt.text(i, history['train_accuracy'][i-1], f'{history["train_accuracy"][i-1]:.2f}', ha='center', va='bottom')
            plt.text(i, history['val_accuracy'][i-1], f'{history["val_accuracy"][i-1]:.2f}', ha='center', va='top')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{dir}/{name}_metrics.png")
    plt.close()  # Close the figure to avoid displaying it in non-interactive environments

    plt.figure(figsize=(20, 10))

    # Plot F1 and F2 scores
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['F1'], label='F1 Score')
    plt.plot(epochs, history['F2'], label='F2 Score')
    plt.title('F1 and F2 score over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    for i in epochs:
        if i % 10 == 0:  # Annotate every 25 epochs
            plt.text(i, history['F1'][i-1], f'{history["F1"][i-1]:.2f}', ha='center', va='bottom')
            plt.text(i, history['F2'][i-1], f'{history["F2"][i-1]:.2f}', ha='center', va='top')
    plt.legend()

    # Plot precision and recall
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['precision'], label='Precision')
    plt.plot(epochs, history['recall'], label='Recall')
    plt.title('Precision and Recall over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    for i in epochs:
        if i % 10 == 0:  # Annotate every 25 epochs
            plt.text(i, history['precision'][i-1], f'{history["precision"][i-1]:.2f}', ha='center', va='bottom')
            plt.text(i, history['recall'][i-1], f'{history["recall"][i-1]:.2f}', ha='center', va='top')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{dir}/{name}_supplementary_metrics.png")
    plt.close()  # Close the figure to avoid displaying it in non-interactive environments





def main():
    # Assuming these paths are correct
    
    parser = argparse.ArgumentParser(description="Run classification model with diff hyper-parameter values")
    
    parser.add_argument("--lr", type=float, required=True, help="learning rate")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--train_path", type=str, required=True, help="train path")
    parser.add_argument("--num", type=int, required=True, help="num")
    

    # Parse the arguments
    args = parser.parse_args()
    
    train_path = args.train_path
    train_dir = f'/home/user1/AlphaGANMRI/{train_path}/Training'
    val_dir = f'/home/user1/AlphaGANMRI/{train_path}/Testing'
    
    lr = args.lr
    batch_size = args.batch_size
    num = args.num
    
    train_transforms = transforms.Compose([
        transforms.Resize(128,128)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        #transforms.GaussianBlur(kernel_size=5,sigma=5.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    

    train_dataset = CustomDataset(directory=train_dir, target_size=(64, 64), class_mode='binary',transform=train_transforms)
    val_dataset = CustomDataset(directory=val_dir, target_size=(64, 64), class_mode='binary',transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 70
    history, conf_matrix = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs)
    
    class_names = ['Not Tumor', 'Tumor'] 
    file_path = f'/home/user1/AlphaGANMRI/test_wo_rotate_classification_models/{lr}_{batch_size}_{train_path}_{num}'
    #save_dir = '/data/user4/Desktop/AlphaGANMRI/v3/test1_classification_models'

    # Save the confusion matrix to a file
    save_confusion_matrix(conf_matrix, class_names, file_path)

    

    
    # Plot and save
    plot_and_save_metrics(history, file_path,f'model_{lr}_{batch_size}_{train_path}_{num}')

    # Save the trained model
    torch.save(model.state_dict(), f'{file_path}/model_{lr}_{batch_size}_{train_path}_{num}.pth')

    print("Training complete.")
    
    

if __name__ == '__main__':
    main()
