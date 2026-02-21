import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

# ---------------- Device ----------------
device = torch.device("cpu")  # PYNQ CPU-only

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

# ---------------- Load Model ----------------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cnn_best.pth", map_location=device))
model.eval()

# ---------------- CIFAR-10 transform ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# ---------------- User Selection ----------------
print("Select input source:")
print("1 - CIFAR-10 test dataset")
print("2 - Camera input (live)")
choice = input("Enter 1 or 2: ")

if choice == "1":
    # ---------------- Load CIFAR-10 ----------------
    testset = torchvision.datasets.CIFAR10(
        root='/home/xilinx/data',  # PYNQ path
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False
    )

    # ---------------- Evaluate ----------------
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

    print("CPU Accuracy on CIFAR-10 test set: {:.2f}%".format(100 * correct / total))

elif choice == "2":
    # ---------------- Camera Input ----------------
    cap = cv2.VideoCapture(0)  # Default camera
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess for CNN
        img = cv2.resize(frame, (32,32))               # Resize to CIFAR-10 size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # OpenCV uses BGR
        img = img.astype(np.float32)/255.0
        img = (img - 0.5)/0.5                          # Normalize
        img = np.transpose(img, (2,0,1))              # HWC -> CHW
        img = np.expand_dims(img, axis=0)             # Add batch dimension
        img_tensor = torch.from_numpy(img).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        # CIFAR-10 classes
        classes = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
        label = classes[predicted.item()]

        # Display
        cv2.putText(frame, f"Predicted: {label}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Camera Input", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid choice. Exiting.")
