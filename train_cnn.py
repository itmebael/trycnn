import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import os
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PechayDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Define classes based on your detection categories
        self.classes = ['Healthy', 'Diseased']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load image paths and labels from directory structure"""
        for class_name in self.classes:
            class_dir = self.data_dir / self.split / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.jpeg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class PechayCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PechayCNN, self).__init__()
        
        # Use pre-trained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
            
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def get_transforms():
    """Define data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the CNN model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Best Val Acc: {best_val_acc:.2f}%')
        print('-' * 50)
        
        scheduler.step()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def evaluate_model(model, test_loader, class_names):
    """Evaluate model and generate detailed metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Generate classification report
    report = classification_report(all_targets, all_predictions, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return report, cm

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    data_dir = "pechay_dataset"  # Change this to your dataset path
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # Create dataset directory structure if it doesn't exist
    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        print(f"Creating dataset directory structure at {data_dir}")
        for split in ['train', 'val', 'test']:
            for class_name in ['Healthy', 'Diseased']:
                (dataset_path / split / class_name).mkdir(parents=True, exist_ok=True)
        print("Please organize your images into the following structure:")
        print("pechay_dataset/")
        print("├── train/")
        print("│   ├── Healthy/")
        print("│   └── Diseased/")
        print("├── val/")
        print("│   ├── Healthy/")
        print("│   └── Diseased/")
        print("└── test/")
        print("    ├── Healthy/")
        print("    └── Diseased/")
        return
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = PechayDataset(data_dir, transform=train_transform, split='train')
    val_dataset = PechayDataset(data_dir, transform=val_transform, split='val')
    test_dataset = PechayDataset(data_dir, transform=val_transform, split='test')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    if len(train_dataset) == 0:
        print("No training data found! Please add images to the dataset directory.")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model
    model = PechayCNN(num_classes=2).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("Starting training...")
    history = train_model(model, train_loader, val_loader, num_epochs, learning_rate)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    if len(test_dataset) > 0:
        print("Evaluating on test set...")
        evaluate_model(model, test_loader, train_dataset.classes)
    
    # Save model
    model_path = f"pechay_cnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': train_dataset.classes,
        'history': history,
        'best_val_acc': history['best_val_acc']
    }, model_path)
    
    print(f"Model saved as: {model_path}")
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")

if __name__ == "__main__":
    main()
