# 1. SVM Implementation
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import pickle
from PIL import Image
import torch

from scripts.Dataset import CustomImageDataset


# Load preprocessed data
def load_preprocessed_data(processed_data_path):
    with open(os.path.join(processed_data_path, 'train_data.pkl'), 'rb') as f:
        train_data, train_labels = pickle.load(f)
    with open(os.path.join(processed_data_path, 'val_data.pkl'), 'rb') as f:
        val_data, val_labels = pickle.load(f)
    scaler = joblib.load(os.path.join(processed_data_path, 'scaler.pkl'))
    return train_data, train_labels, val_data, val_labels, scaler

# Load preprocessed train and validation data
processed_data_path = '../processed_data'
print("Loading preprocessed training and validation data...")
train_data, train_labels, val_data, val_labels, scaler = load_preprocessed_data(processed_data_path)

# Select only 2000 samples (1000 cats and 1000 dogs) for training
cat_indices = [i for i, label in enumerate(train_labels) if label == 0][:1000]
dog_indices = [i for i, label in enumerate(train_labels) if label == 1][:1000]
selected_indices = cat_indices + dog_indices
train_data = train_data[selected_indices]
train_labels = train_labels[selected_indices]

print("Data loading and selection completed.")

# Train SVM model on GPU using cuML
clf = svm.SVC(kernel='linear', probability=True)
print("Begin training svm...")
clf.fit(train_data, train_labels)

# Evaluate SVM
val_preds = clf.predict(val_data)
print(f"SVM Validation Accuracy: {accuracy_score(val_labels, val_preds):.4f}")

# Confusion Matrix for SVM
cm_svm = confusion_matrix(val_labels, val_preds)
ConfusionMatrixDisplay(cm_svm, display_labels=['Cat', 'Dog']).plot()
plt.title("SVM Confusion Matrix")
plt.show()

# Save SVM Model
joblib.dump(clf, '../results/svm_model.pkl')