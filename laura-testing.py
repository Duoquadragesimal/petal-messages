import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms as transforms
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data_path = "dataset" # Path to your dataset
image_list = []
label_list = []

class_list_ = ["Not origami flower","Origami Flower"] # Our list of classifications

for category in class_list_:
    for image in os.listdir(f"{train_data_path}{category}"):
        image_list.append(f"{train_data_path}{category}/{image}")
        label_list.append(f"{category}")

df = pd.DataFrame()

df["image"] = image_list
df["label"] = label_list

ratio = 0.50 #50% of data will be for testing
train_df, test_df = train_test_split(df, test_size=ratio, stratify=df["label"], random_state=42)

image_size = 28

training_transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                            transforms.RandomRotation(10),
                                             transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

BATCH_SIZE = 10

class CustomTrainingData(Dataset):
    def __init__(self, csv_df, class_list, transform=None):
        self.df = csv_df
        self.transform = transform
        self.class_list = class_list
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        try:
            image_path = self.df.iloc[index]["image"]
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return self.__getitem__((index + 1) % len(self.df)) 
        label = self.class_list.index(self.df.iloc[index]["label"])
        if self.transform:
            image = self.transform(image)
        return image, label
    
train_data_object = CustomTrainingData(train_df, class_list_, training_transform)
test_data_object = CustomTrainingData(test_df, class_list_, test_transform)
train_loader = DataLoader(train_data_object, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data_object, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)