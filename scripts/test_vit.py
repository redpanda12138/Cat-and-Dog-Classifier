from datasets import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig

from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from natsort import natsorted
import torch
import os

config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)  # Assuming binary classification
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

class CustomTestDataset(TorchDataset):
    def __init__(self, test_path, transform=None):
        self.test_path = test_path
        self.img_names = natsorted(os.listdir(test_path))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.test_path, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {
            'pixel_values': img,
            'label': img_name  # 使用图像名作为标签（可以根据实际情况调整）
        }

# 定义数据转换
transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# 使用自定义的数据集
test_dataset = CustomTestDataset('../data/test', transform=transform_vit)

# 使用 DataLoader 分批加载测试数据
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 使用模型进行预测并保存结果
model.load_state_dict(torch.load('../results/vit_model/vit_model_epoch_30.pth'))
model.eval()
predictions = []
labels = []

with torch.no_grad():
    progress_bar = tqdm(test_loader, desc="Generating Predictions on Test Set", unit="batch")

    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)  # 移动到 GPU 或 CPU
        outputs = model(pixel_values)
        _, predicted = torch.max(outputs.logits, 1)

        predictions.extend(predicted.cpu().numpy())
        labels.extend(batch['label'])  # 保存图像标签

# 将结果保存到 CSV
import pandas as pd

results_df = pd.DataFrame({
    'ImageName': labels,
    'PredictedLabel': predictions
})
results_df.to_csv('../results/test_results.csv', index=False)
print("Test predictions saved to test_results.csv")

