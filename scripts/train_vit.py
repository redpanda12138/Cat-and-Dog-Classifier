# 3. Vision Transformer (ViT) Implementation (using Hugging Face Transformers)
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments, DefaultDataCollator, ViTConfig, logging
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset
import pandas as pd
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

from scripts.Dataset import CustomImageDataset

# Training setup for multiple learning rates
learning_rates = [1e-3, 5e-4]

# Load ViT model and feature extractor
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training ViT model with device: {device}')

vit_results = {}
logging.set_verbosity_error()

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

# 创建 Hugging Face Dataset 格式的数据集
def create_huggingface_dataset(image_folder_dataset, transform):
    pixel_values = []
    labels = []
    for image_path, label in image_folder_dataset.samples:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img)  # 这将是一个 torch.Tensor
        img_list = img_tensor.numpy().tolist()  # 转换为可序列化的列表
        pixel_values.append(img_list)
        labels.append(label)

    # 使用 from_dict 创建数据集
    return Dataset.from_dict({
        'pixel_values': pixel_values,
        'label': labels
    })

for lr in learning_rates:
    config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)  # 若为二分类
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
    model = model.to(device)

    # Prepare data
    transform_vit = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])

    # train_dataset_vit = datasets.ImageFolder('../data/train', transform=transform_vit)
    # val_dataset_vit = datasets.ImageFolder('../data/val', transform=transform_vit)
    # test_images = load_test_images('../data/test', transform_vit)
    # test_dataset_vit = CustomImageDataset(test_images)  # Assuming CustomImageDataset only takes one argument

    # 使用函数创建训练和验证数据集
    train_dataset_vit = create_huggingface_dataset(
        datasets.ImageFolder('../data/train'), transform_vit
    )
    val_dataset_vit = create_huggingface_dataset(
        datasets.ImageFolder('../data/val'), transform_vit
    )
    # 测试数据集
    test_images = load_test_images('../data/test', transform_vit)
    test_data = [{'pixel_values': img, 'label': idx} for idx, img in enumerate(test_images)]
    test_dataset_vit = Dataset.from_pandas(pd.DataFrame(test_data))

    # Select only 2000 samples (1000 cats and 1000 dogs) for training
    cat_indices = [i for i, (path, label) in enumerate(train_dataset_vit.samples) if label == 0][:1000]
    dog_indices = [i for i, (path, label) in enumerate(train_dataset_vit.samples) if label == 1][:1000]
    selected_indices = cat_indices + dog_indices
    train_dataset_vit.samples = [train_dataset_vit.samples[i] for i in selected_indices]

    data_collator = DefaultDataCollator(return_tensors="pt")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="../results",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        learning_rate=lr,
        save_strategy="epoch",
        logging_dir="../logs",
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_vit,
        eval_dataset=val_dataset_vit,
        data_collator=data_collator,
    )

    # Train ViT
    trainer.train()
    print(f'Training ViT model with learning rate: {lr}')

    # Save ViT Model
    trainer.save_model(output_dir=f'../results/vit_model_lr_{lr}')

    # Evaluate on Test Set
    test_loader_vit = DataLoader(test_dataset_vit, batch_size=16, shuffle=False)
    test_preds_vit, test_labels_vit = [], []
    model.eval()
    with torch.no_grad():
        for images in test_loader_vit:
            images = images.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            test_preds_vit.extend(predicted.cpu().numpy())

    # Save Test Results
    test_results_vit_df = pd.DataFrame({'PredictedLabel': test_preds_vit})
    test_results_vit_df.to_csv(f'../results/vit_test_results_lr_{lr}.csv', index=False)

    # Save training and evaluation metrics for plotting
    vit_results[lr] = {
        'train_losses': [entry['loss'] for entry in trainer.state.log_history if 'loss' in entry],
        'val_losses': [entry['eval_loss'] for entry in trainer.state.log_history if 'eval_loss' in entry],
        'train_accuracies': [entry.get('accuracy', 0) for entry in trainer.state.log_history if 'accuracy' in entry],
        'val_accuracies': [entry.get('eval_accuracy', 0) for entry in trainer.state.log_history if 'eval_accuracy' in entry]
    }

# Plotting the results
plt.figure(figsize=(15, 10))

# ViT Results Plot
for lr in learning_rates:
    plt.subplot(2, 2, 1)
    plt.plot(vit_results[lr]['train_losses'], label=f"Train Loss (lr={lr})")
    plt.title("VIT Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(vit_results[lr]['val_losses'], label=f"Val Loss (lr={lr})")
    plt.title("VIT Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(vit_results[lr]['train_accuracies'], label=f"Train Accuracy (lr={lr})")
    plt.title("VIT Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(vit_results[lr]['val_accuracies'], label=f"Val Accuracy (lr={lr})")
    plt.title("VIT Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

plt.tight_layout()
plt.show()
