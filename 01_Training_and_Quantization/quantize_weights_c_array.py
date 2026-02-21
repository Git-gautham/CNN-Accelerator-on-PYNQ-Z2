import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

# ---------------- Settings ----------------
DEVICE = torch.device("cpu")  # PYNQ uses CPU to load weights
FIXED_POINT_FRACTIONAL_BITS = 10  # ap_fixed<16,6> equivalent
OUTPUT_FILE = "quantized_weights.txt"

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
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load("cnn_best.pth", map_location=DEVICE))
model.eval()

# ---------------- Quantization Function ----------------
def quantize(arr, fractional_bits=FIXED_POINT_FRACTIONAL_BITS):
    scale = 2 ** fractional_bits
    return np.round(arr * scale) / scale

def quantize_model(model):
    for name, param in model.named_parameters():
        param.data = torch.from_numpy(quantize(param.data.cpu().numpy())).to(param.device)

# Apply quantization
quantize_model(model)

# ---------------- Optional: Test Accuracy ----------------
def test_accuracy(model, batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Quantized model accuracy: {100 * correct / total:.2f}%")

# Uncomment below to test accuracy
# test_accuracy(model)

# ---------------- Export Quantized Weights to TXT ----------------
np.set_printoptions(threshold=np.inf, precision=8, floatmode='fixed', linewidth=200)

def export_quantized_weights_to_file(layer, name, f):
    w = layer.weight.data.cpu().numpy()
    b = layer.bias.data.cpu().numpy()
    w_str = np.array2string(w.flatten(), separator=', ')
    b_str = np.array2string(b.flatten(), separator=', ')
    f.write(f"// Layer: {name}\n")
    f.write(f"{name}_w = {w_str}\n")
    f.write(f"{name}_b = {b_str}\n\n")

with open(OUTPUT_FILE, "w") as f:
    export_quantized_weights_to_file(model.conv1, "conv1", f)
    export_quantized_weights_to_file(model.conv2, "conv2", f)
    export_quantized_weights_to_file(model.conv3, "conv3", f)
    export_quantized_weights_to_file(model.fc, "fc", f)

print(f"All quantized weights and biases saved to {OUTPUT_FILE}")