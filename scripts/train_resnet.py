# 2. ResNet Implementation (using PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd

from scripts.Dataset import CustomImageDataset


# Load test dataset without requiring class folders
def load_test_images(test_path, transform):
    test_images = []
    img_names = sorted(os.listdir(test_path))
    for img_name in img_names:
        img_path = os.path.join(test_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # 将 numpy.ndarray 转换为 PIL 图像
        if transform:
            img = transform(img)
        test_images.append(img)
    return test_images

# Prepare data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder('../data/train', transform=transform)
val_dataset = datasets.ImageFolder('../data/val', transform=transform)
test_images = load_test_images('../data/test', transform)

# Select only 2000 samples (1000 cats and 1000 dogs) for training
cat_indices = [i for i, (path, label) in enumerate(train_dataset.samples) if label == 0][:1000]
dog_indices = [i for i, (path, label) in enumerate(train_dataset.samples) if label == 1][:1000]
selected_indices = cat_indices + dog_indices
train_dataset.samples = [train_dataset.samples[i] for i in selected_indices]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = CustomImageDataset(test_images)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training setup for multiple learning rates
learning_rates = [1e-3, 5e-4]
resnet_results = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for lr in learning_rates:
    # Load pretrained ResNet model
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet = resnet.to(device)

    # Optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=lr)

    # Training and validation metrics storage
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Train ResNet
    for epoch in range(5):
        resnet.train()
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/5", unit="batch")
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1), 'accuracy': 100 * correct / total})

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

        # Validate ResNet
        resnet.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = resnet(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)

    resnet_results[lr] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

    # Save ResNet Model Checkpoint
    torch.save(resnet.state_dict(), f'../results/resnet_model_lr_{lr}.pth')

    # Evaluate on Test Set
    resnet.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.numpy())


    # Save Test Results
    test_results_df = pd.DataFrame({'TrueLabel': test_labels, 'PredictedLabel': test_preds})
    test_results_df.to_csv(f'../results/resnet_test_results_lr_{lr}.csv', index=False)

# Plotting the results
plt.figure(figsize=(15, 10))

# ResNet Results Plot
for lr in learning_rates:
    plt.subplot(2, 2, 1)
    plt.plot(resnet_results[lr]['train_losses'], label=f"Train Loss (lr={lr})")
    plt.title("ResNet Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(resnet_results[lr]['val_losses'], label=f"Val Loss (lr={lr})")
    plt.title("ResNet Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(resnet_results[lr]['train_accuracies'], label=f"Train Accuracy (lr={lr})")
    plt.title("ResNet Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(resnet_results[lr]['val_accuracies'], label=f"Val Accuracy (lr={lr})")
    plt.title("ResNet Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

plt.tight_layout()
plt.show()
