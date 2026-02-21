#16 bit fixed point quantized weights and bias in folder for c++ test 
import torch
import torch.nn as nn
import numpy as np
import os

# ---------------- Settings ----------------
DEVICE = torch.device("cpu")
FIXED_POINT_FRACTIONAL_BITS = 10  # ap_fixed<16,6> equivalent
OUTPUT_DIR = "quantized_weights_files"

# Create folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    def forward(self, x):
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

# ---------------- Quantization ----------------
def quantize(arr, fractional_bits=FIXED_POINT_FRACTIONAL_BITS):
    scale = 2 ** fractional_bits
    return np.round(arr * scale) / scale

def quantize_model(model):
    for name, param in model.named_parameters():
        param.data = torch.from_numpy(quantize(param.data.cpu().numpy())).to(param.device)

quantize_model(model)

# ---------------- Save each layer as separate file ----------------
def save_layer(layer, name):
    w = layer.weight.data.cpu().numpy().flatten()
    b = layer.bias.data.cpu().numpy().flatten()
    np.savetxt(os.path.join(OUTPUT_DIR, f"{name}.weight.txt"), w, fmt='%.8f', delimiter=', ')
    np.savetxt(os.path.join(OUTPUT_DIR, f"{name}.bias.txt"), b, fmt='%.8f', delimiter=', ')

save_layer(model.conv1, "conv1")
save_layer(model.conv2, "conv2")
save_layer(model.conv3, "conv3")
save_layer(model.fc,    "fc")

print(f"All quantized weights and biases saved in {OUTPUT_DIR} as separate .txt files.")