import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# ---------------------------------------------------------
# 1. Device Configuration
# ---------------------------------------------------------
# Select GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. Model Architecture
# ---------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    A basic Convolutional Neural Network (CNN) for CIFAR-10.
    Architecture: 3 Conv layers with ReLU and MaxPool, followed by 1 Linear layer.
    """
    def __init__(self):
        super().__init__()
        # Input: 3x32x32
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # After three 2x2 pooling layers, image size is 32/2/2/2 = 4x4
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        # Feature extraction
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize model and move to target device
model = SimpleCNN().to(device)

# ---------------------------------------------------------
# 3. Data Preparation
# ---------------------------------------------------------
# Convert images to tensors and normalize to range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load CIFAR-10 training/testing sets
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# ---------------------------------------------------------
# 4. Training Hyperparameters & Setup
# ---------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
best_acc = 0

# ---------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        # Optimization step
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluation phase
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1:02d}, Loss: {running_loss:.3f}, Test Acc: {acc:.2f}%")

    # Save the model state if it achieves the highest accuracy so far
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "cnn_best.pth")

print(f"Training complete. Best accuracy: {best_acc:.2f}%")

# ---------------------------------------------------------
# 6. Post-Training: Exporting Weights & Verification
# ---------------------------------------------------------
# Load the weights from the best performing epoch
model.load_state_dict(torch.load("cnn_best.pth"))
model.eval()

# Export weights to text files for external verification/deployment
state = model.state_dict()
for name, param in state.items():
    # Flatten weights to 1D for simpler .txt storage
    np.savetxt(f"{name}.txt", param.cpu().numpy().flatten())

print("Weights exported.")

# Export the first image of the test set for verification
img, label = testset[0]
print(f"True label of first test image: {label}")

np.savetxt("image.txt", img.numpy().flatten())
print("Image exported.")

# Run an inference check on the exported image
img_batch = img.unsqueeze(0).to(device)
out = model(img_batch)
print(f"Python prediction for exported image: {out.argmax().item()}")