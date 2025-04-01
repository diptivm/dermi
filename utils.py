import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from tqdm import tqdm


class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, blackhat_threshold, kernel_size_hair, kernel_size_blur, blur_func, doBlur, transform=None,):
        self.dataframe = dataframe
        self.transform = transform
        self.blackhat_threshold = blackhat_threshold
        self.kernel_size_tuple = (kernel_size_hair, kernel_size_hair)
        self.kernel_size_blur = kernel_size_blur
        self.blur_func = blur_func
        self.doBlur = doBlur

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        # image = Image.open(img_path).convert('RGB')
        image = cv2.imread(img_path)
        label = int(self.dataframe.iloc[idx, 1])
        cleaned_image = remove_hair(image, kernel_size_tuple=self.kernel_size_tuple, blackhat_threshold=self.blackhat_threshold)
        if(self.doBlur):
          cleaned_image = blurring(cleaned_image, kernel_size=self.kernel_size_blur,blur_func=self.blur_func)

        cleaned_image = Image.fromarray(cleaned_image)

        if self.transform:
            cleaned_image = self.transform(cleaned_image)

        return cleaned_image, label
    

class EfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB3, self).__init__()
        # Load pretrained EfficientNet B3
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        # Get the number of input features for the classifier layer
        in_features = self.model.classifier[1].in_features
        # Replace the final classification layer
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        # Softmax for output probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x


def remove_hair(image, kernel_size_tuple=(15,15), blackhat_threshold=18):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a blackhat filter to find the hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size_tuple)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold the blackhat image
    thresh = cv2.threshold(blackhat, blackhat_threshold, 255, cv2.THRESH_BINARY)[1]

    # Inpaint the original image based on the thresholded image
    inpainted_image = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    return inpainted_image


def blurring(image, kernel_size, blur_func=cv2.medianBlur, sigma=None):
  if blur_func not in (cv2.medianBlur, cv2.GaussianBlur):
    raise ValueError("blur_func must be one of cv2.medianBlur or cv2.GaussianBlur")
  if blur_func == cv2.GaussianBlur:
    blur_args = (image, (kernel_size, kernel_size), sigma)
  else:
    blur_args = (image, kernel_size)
  blurred_img = blur_func(*blur_args)
  return blurred_img
    

def train_one_epoch(epoch, model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    correct_train = 0

    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", ncols=100):
        images, labels = images.to(device, dtype=torch.float32), labels.cuda().long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = correct_train / len(train_loader.dataset)

    return train_loss, train_accuracy


def validate_one_epoch(epoch, model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_val = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", ncols=100):
            images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct_val / len(val_loader.dataset)

    return val_loss, val_accuracy


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, early_stopping_patience, device):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validate_one_epoch(epoch, model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping")
            break

    return train_losses, train_accuracies, val_losses, val_accuracies, best_val_loss