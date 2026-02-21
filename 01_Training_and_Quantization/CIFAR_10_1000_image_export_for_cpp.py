#code to extract 1000 images

import numpy as np
from torchvision import datasets, transforms

# Same normalization as training
transform = transforms.Compose([
    transforms.ToTensor(),
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Select first 1000 images
num_images = 1000
with open("cifar_test_1000.txt", "w") as f:
    for i in range(num_images):
        img, label = testset[i]
        f.write(f"{label} ")
        for pixel in img.numpy().flatten():
            f.write(f"{pixel} ")
        f.write("\n")
