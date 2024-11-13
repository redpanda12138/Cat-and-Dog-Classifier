import torch
from torchvision import datasets, transforms
from datasets import Dataset
from PIL import Image
import os

from transformers import ViTImageProcessor

feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# 创建并分批保存 Hugging Face Dataset
def create_and_save_huggingface_dataset(image_folder_dataset, transform, save_path, max_samples=10000, batch_size=5000):
    pixel_values = []
    labels = []
    cat_count, dog_count = 0, 0
    batch_count = 0

    for image_path, label in image_folder_dataset.samples:
        if label == 0 and cat_count < max_samples // 2:  # 猫类样本数量
            cat_count += 1
        elif label == 1 and dog_count < max_samples // 2:  # 狗类样本数量
            dog_count += 1
        else:
            continue

        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img)  # 转换为 torch.Tensor
        pixel_values.append(img_tensor.numpy())  # 保存为 NumPy 数组，减少内存占用
        labels.append(label)

        # 当达到 batch_size 时，保存当前批次
        if len(pixel_values) >= batch_size:
            dataset = Dataset.from_dict({'pixel_values': pixel_values, 'label': labels})
            batch_save_path = f"{save_path}_batch_{batch_count}"
            dataset.save_to_disk(batch_save_path)
            print(f"Batch {batch_count} saved to {batch_save_path}")

            # 清空列表以释放内存
            pixel_values = []
            labels = []
            batch_count += 1

        # 如果达到了最大样本数则退出循环
        if cat_count + dog_count >= max_samples:
            break


# 数据变换
transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# 创建并保存训练数据集（分批处理）
val_image_folder = datasets.ImageFolder('../data/val')
create_and_save_huggingface_dataset(val_image_folder, transform_vit, '../processed_data/vit_val_data', max_samples=10000, batch_size=5000)
train_image_folder = datasets.ImageFolder('../data/train')
create_and_save_huggingface_dataset(train_image_folder, transform_vit, '../processed_data/vit_train_data', max_samples=10000, batch_size=5000)
