import gc

from matplotlib import pyplot as plt
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset
import cv2
import os
from PIL import Image
import torch.nn.functional as F

# Load ViT model and feature extractor
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training ViT model with device: {device}')


# Load test dataset without requiring class folders
def load_test_images(test_path, transform):
    test_images = []
    test_labels = []
    img_names = sorted(os.listdir(test_path))
    for img_name in img_names:
        img_path = os.path.join(test_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # 将 numpy.ndarray 转换为 PIL 图像
        if transform:
            img = transform(img)
        test_images.append(img)
        test_labels.append(img_name)  # 将图像名称作为标签
    return test_images, test_labels

# 绘制测试结果图表
def save_test_results(avg_test_loss, test_accuracy, output_path):
    # 创建一个图表
    fig, ax = plt.subplots(figsize=(10, 5))

    # 绘制测试损失和准确率
    metrics = ['Test Loss', 'Test Accuracy']
    values = [avg_test_loss, test_accuracy]

    # 绘制柱状图
    ax.bar(metrics, values, color=['skyblue', 'salmon'])
    ax.set_title('Test Results')
    ax.set_ylabel('Values')
    ax.set_ylim(0, max(values) + 5)  # 设置 y 轴范围

    # 在每个柱状图顶部添加数值标签
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, f'{v:.2f}', ha='center', fontsize=12)

    # 保存图表
    plt.savefig(output_path)
    plt.close()
    print(f"Test results saved as a chart in: {output_path}")

# Prepare data
transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])


# 使用函数创建训练和验证数据集
train_dataset_vit = Dataset.load_from_disk('../processed_data/vit_train_data', keep_in_memory=False)
val_dataset_vit = Dataset.load_from_disk('../processed_data/vit_val_data', keep_in_memory=False)

train_dataset_vit.set_format(type='torch', columns=['pixel_values', 'label'])
val_dataset_vit.set_format(type='torch', columns=['pixel_values', 'label'])

# 创建数据加载器
def collate_fn(batch):
    # 将数据转换为 PyTorch 张量
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }

# 使用自定义 collate_fn 来适配 DataLoader
train_loader = DataLoader(train_dataset_vit, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(train_dataset_vit, batch_size=4, shuffle=False, collate_fn=collate_fn)

# # 测试数据集
# test_images, test_labels = load_test_images('../data/test', transform_vit)
# # 将数据转换为可以被 Hugging Face Dataset 使用的格式
# test_data = [
#     {
#         'pixel_values': [img.numpy().tolist() for img in test_images],  # 将张量转换为列表
#         'label': test_labels  # 假设这里使用 idx 作为标签，根据需要进行调整
#     }
#     for img in test_images
# ]
# test_dataset_vit = Dataset.from_dict(test_data)

# Training parameters
learning_rate = 1e-4
num_epochs = 30
config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)  # Assuming binary classification
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

accumulation_steps = 4
# Training loop with tqdm
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch",
                        total=len(train_loader))

    for i, batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        # Zero the parameter gradients every accumulation_steps batches
        if i % accumulation_steps == 0:
            optimizer.zero_grad()

        # Forward pass
        outputs = model(pixel_values)
        loss = F.cross_entropy(outputs.logits, labels)

        # Backward pass (loss / accumulation_steps)
        (loss / accumulation_steps).backward()

        # Update weights after accumulation_steps batches
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

    # Save ViT Model after each epoch
    model_save_path = f'../results/vit_model/vit_model_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Validation step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values)
            loss = F.cross_entropy(outputs.logits, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    save_test_results(avg_val_loss, val_accuracy, '../figure/vit_val_results.png')
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    torch.cuda.empty_cache()
    gc.collect()

print('Training complete.')
