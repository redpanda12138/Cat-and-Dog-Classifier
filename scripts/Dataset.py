import torch
from torch.utils.data import Dataset

# 自定义 Dataset 类
class CustomImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(idx)  # 使用 -1 作为占位符标签