# -----------------------------
# PYNQ CPU-Friendly CNN Notebook
# -----------------------------
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from IPython.display import display, clear_output
import platform
import time

# ---------------- Device ----------------
device = torch.device("cpu")  # CPU only for PYNQ

# ---------------- Model ----------------
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

# ---------------- Load trained model ----------------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cnn_best.pth", map_location=device))
model.eval()

# ---------------- CIFAR-10 transform ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# ---------------- Classes ----------------
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# ---------------- Dataset path ----------------
if platform.system() == "Windows":
    data_path = "./data"  # Windows-friendly
else:
    data_path = "/home/xilinx/data"  # PYNQ path

# ---------------- User choice ----------------
print("Select input source:")
print("1 - CIFAR-10 test dataset (first 100 images)")
print("2 - USB/Camera input (live)")
choice = input("Enter 1 or 2: ")

# ---------------- CIFAR-10 option ----------------
if choice == "1":
    testset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )
    # Limit to first 100 images for PYNQ
    testset = torch.utils.data.Subset(testset, list(range(100)))
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=10,
        shuffle=False
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"CPU Accuracy on 100 CIFAR-10 test images: {100*correct/total:.2f}%")

# ---------------- Camera option ----------------
elif choice == "2":
    cap = cv2.VideoCapture(0)
    print("Press Ctrl+C to stop the live feed")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Preprocess for CNN
            img = cv2.resize(frame, (32,32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)/255.0
            img = (img - 0.5)/0.5
            img = np.transpose(img, (2,0,1))
            img = np.expand_dims(img, axis=0)
            img_tensor = torch.from_numpy(img).to(device)

            # CNN inference
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                label = classes[predicted.item()]

            # Display inline in Jupyter Notebook
            img_display = frame.copy()
            img_display = cv2.putText(img_display, f"Predicted: {label}", (10,30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            img_pil = Image.fromarray(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
            clear_output(wait=True)
            display(img_pil)

            time.sleep(0.1)  # adjust for CPU speed

    except KeyboardInterrupt:
        print("Live camera feed stopped by user")
    finally:
        cap.release()

else:
    print("Invalid choice. Exiting.")
