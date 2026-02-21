# -*- coding: utf-8 -*-
"""
CIFAR10 Test + Export Image for C++
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model Definition (MUST match C++)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -----------------------------
# Load Model
# -----------------------------
model = SimpleCNN().to(device)

# load your trained weights here
model.load_state_dict(torch.load("cnn_best.pth", map_location=device))

model.eval()

# -----------------------------
# Load CIFAR-10 Test Set
# -----------------------------
transform = transforms.ToTensor()

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# -----------------------------
# Choose Different Image
# -----------------------------
index = 25  # Change this number to test different images

image, label = testset[index]

print("True label:", label)
print("Image shape:", image.shape)

# -----------------------------
# Save Image for C++
# -----------------------------
np.savetxt("image.txt", image.numpy().flatten())
print("image.txt saved for C++")

# -----------------------------
# Python Prediction
# -----------------------------
with torch.no_grad():
    output = model(image.unsqueeze(0).to(device))
    prediction = torch.argmax(output, dim=1)

print("Python prediction:", prediction.item())
