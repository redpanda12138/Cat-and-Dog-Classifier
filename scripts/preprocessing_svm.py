# Python script for data preprocessing for cat vs dog classifier
# This script will save preprocessed data for later use in training models

from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
import os
import joblib
import pickle

# Prepare data
def load_data(data_path, sample_fraction=0.8):
    data, labels = [], []
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        label = 0 if class_name == 'cat' else 1
        img_names = os.listdir(class_path)
        img_names = img_names[:int(len(img_names) * sample_fraction)]  # Use a fraction of data for preprocessing
        for idx, img_name in enumerate(img_names):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            data.append(img.flatten())
            labels.append(label)
            if idx % 100 == 0:
                print(f"Processed {idx} images from class '{class_name}'")
    return np.array(data), np.array(labels)

# Load train and validation data
print("Loading training data...")
train_data, train_labels = load_data('../data/train')
print("Loading validation data...")
val_data, val_labels = load_data('../data/val')

# Standardize data
print("Standardizing data...")
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)

# Create directory for processed data if it doesn't exist
processed_data_path = './processed_data'
os.makedirs(processed_data_path, exist_ok=True)

# Save preprocessed data and scaler
print("Saving preprocessed data...")
with open(os.path.join(processed_data_path, 'train_data.pkl'), 'wb') as f:
    pickle.dump((train_data, train_labels), f)

with open(os.path.join(processed_data_path, 'val_data.pkl'), 'wb') as f:
    pickle.dump((val_data, val_labels), f)

joblib.dump(scaler, os.path.join(processed_data_path, 'scaler.pkl'))

print("Data preprocessing completed and saved.")
