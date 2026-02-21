# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 14:39:27 2026

@author: gauth
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import numpy as np

# Load image.txt
img = np.loadtxt("image.txt")

# Convert to tensor
image = torch.tensor(img, dtype=torch.float32)

# Reshape to CHW (3,32,32)
image = image.view(3, 32, 32)

print("Image shape:", image.shape)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64*4*4,10)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN().to(device)
model.eval()

with torch.no_grad():
    out = model(image.unsqueeze(0))
    pred = torch.argmax(out, dim=1)

print("Python prediction:", pred.item())
