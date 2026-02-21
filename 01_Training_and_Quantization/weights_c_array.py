#Code to export weights as c arrays 
import torch
import numpy as np

device = torch.device("cpu")

# Load model
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,16,3,padding=1)
        self.conv2 = torch.nn.Conv2d(16,32,3,padding=1)
        self.conv3 = torch.nn.Conv2d(32,64,3,padding=1)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(64*4*4,10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cnn_best.pth", map_location=device))
model.eval()

# ---------------- Disable truncation ----------------
np.set_printoptions(threshold=np.inf, precision=8, floatmode='fixed', linewidth=200)

# ---------------- Save to file ----------------
def export_weights_to_file(layer, name, f):
    w = layer.weight.data.cpu().numpy()
    b = layer.bias.data.cpu().numpy()

    w_str = np.array2string(w.flatten(), separator=', ')
    b_str = np.array2string(b.flatten(), separator=', ')

    f.write(f"// Layer: {name}\n")
    f.write(f"const data_t {name}_w[] = {w_str};\n")
    f.write(f"const data_t {name}_b[] = {b_str};\n\n")

with open("cnn_weights.h", "w") as f:
    export_weights_to_file(model.conv1, "conv1", f)
    export_weights_to_file(model.conv2, "conv2", f)
    export_weights_to_file(model.conv3, "conv3", f)
    export_weights_to_file(model.fc, "fc", f)

print("All weights and biases saved to cnn_weights.h without truncation!")