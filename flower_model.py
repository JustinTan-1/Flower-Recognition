import torch 
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from Recognition import Recognition2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

import scipy.io as sio


class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx])
        image = self.images[idx]
        return image, label
    def __len__(self):
        return len(self.labels)
    
number_to_label = {}
classNames = []
images = []
labels = []
counter = 1
# Loading classnames
with open("./flowers/labels.txt") as f:
    label = f.readline()
    while label:
        trimmed_label = label.strip("\'\n ")
        number_to_label[counter] = trimmed_label
        classNames.append(trimmed_label)
        label = f.readline()
        counter += 1

transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.RandomHorizontalFlip(),
    #transforms.Grayscale(num_output_channels=1),
    #transforms.Resize((128, 128)),
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading images and labels together
encoder = LabelEncoder()
encoder.fit(classNames)
folder = "./flowers/jpg"
mat_data = sio.loadmat("./flowers/imagelabels")
i = 0
for image in os.listdir(folder):
    processed_image = cv2.imread(os.path.join(folder, image))
    if len(processed_image) != 0:
        print(image)
        normalized_tensor = transform(processed_image)
        #plt.imshow(normalized_tensor)
        #plt.show()
        images.append(normalized_tensor)
        labels.append(mat_data["labels"][0][i] - 1)
    i += 1

print(classNames[101])


# Creating datasets and dataloaders for the images
dataset = custom_dataset(images, labels)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = Recognition2()
"""
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, out_features=102)
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True
"""
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

rounds = 30
for epoch in range(rounds):
    total = 0
    correct = 0
    total_loss = 0.0
    for image, label in train_loader:
        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()    
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader)}, Accuracy: {correct * 100 / total}%")

correct = 0
total = 0
model.eval()

with torch.no_grad():
    for image, label in test_loader:
        outputs = model(image)
        _, prediction = torch.max(outputs, 1)
        total += label.size(0)
        correct += (prediction == label).sum().item() # Compares Tensors True = 1 and False = 0 then sums up all the trues
print(f"Accuracy: {100 * correct/total}")
#torch.save(model.state_dict(), "./saved_model/model.pt")