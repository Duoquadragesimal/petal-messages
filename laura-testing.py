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
    for image in os.listdir(os.path.join(train_data_path, category)):
        image_list.append(os.path.join(train_data_path, category, image))
        label_list.append(category)

print("Total images found:", len(image_list))
print("Total labels found:", len(label_list))

print("\nImages per category:")
for category in class_list_:
    print(category, ":", label_list.count(category))

df = pd.DataFrame()

df["image"] = image_list
df["label"] = label_list

ratio = 0.50 #50% of data will be for testing
train_df, test_df = train_test_split(df, test_size=ratio, stratify=df["label"], random_state=42)

print("\nTraining images:", len(train_df))
print("Testing images:", len(test_df))

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

images, labels = next(iter(train_loader))

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.counter = 0
        self.best_model_state = None
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()

        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, input_size=(image_size, image_size), channels=3): 
        super(ImageClassifier, self).__init__()

        self.input_size = input_size
        self.channels = channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Calculate the size of the flattened features
        self._to_linear = None
        self._calculate_to_linear(input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def _calculate_to_linear(self, input_size):
        # This function calculates the size of the flattened features
        x = torch.randn(1, self.channels, *input_size)
        self.conv_forward(x)

    def conv_forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.conv_forward(x)

        # Flatten the output for the fully connected layer
        x = x.view(-1, self._to_linear)

        # Fully connected layers with ReLU and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = len(class_list_)
INPUT_SIZE = (image_size, image_size)
CHANNELS = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# INIT MODEL
model = ImageClassifier(NUM_CLASSES, INPUT_SIZE, CHANNELS).to(device)

# LOSS FUNCTION
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

early_stopping = EarlyStopping()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}')
    
    # Validation
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        early_stopping(running_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    early_stopping.load_best_model(model)
    
    # Calculate metrics
    accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    # Print metrics
    print(f'Epoch [{epoch+1}/{EPOCHS}]')
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

print('Training finished!')

# Save the model
torch.save(model.state_dict(), 'classifier.pth')

# LOAD THE MODEL

model = ImageClassifier(
    num_classes=len(class_list_),
    input_size=(image_size, image_size)
).to(device)

model.load_state_dict(torch.load('classifier.pth', map_location=device))
model.eval()

# PREPARE THE IMAGE

img = Image.open('test.jpg').convert('RGB')

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_tensor = transform(img).unsqueeze(0).to(device)

# MAKE PREDICTION

with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    prediction = class_list_[predicted.item()]

print(f"Predicted class: {prediction}")
