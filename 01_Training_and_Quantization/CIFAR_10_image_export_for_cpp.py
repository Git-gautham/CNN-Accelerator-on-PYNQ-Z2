# -*- coding: utf-8 -*-
"""
Test 10 CIFAR10 images and export for C++
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
# Model Definition
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
# CIFAR-10 Class Names
# -----------------------------
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("\nTesting 10 images...\n")

# -----------------------------
# Loop Over 10 Images
# -----------------------------
for i in range(10):

    image, label = testset[i]

    # Save image for C++
    filename = f"image_{i}.txt"
    np.savetxt(filename, image.numpy().flatten())

    # Python prediction
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        pred = torch.argmax(output, dim=1).item()

    print(f"Image {i}")
    print("True label:", label, "-", classes[label])
    print("Python prediction:", pred, "-", classes[pred])
    print("Saved as:", filename)
    print("-" * 40)
