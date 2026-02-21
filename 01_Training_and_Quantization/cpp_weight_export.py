# Save weights to txt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model.load_state_dict(torch.load("cnn_best.pth", map_location=device))
model.eval()
def save_tensor(tensor, filename):
    np.savetxt(filename, tensor.detach().cpu().numpy().flatten())

save_tensor(model.conv1.weight, "conv1.weight.txt")
save_tensor(model.conv1.bias,   "conv1.bias.txt")
save_tensor(model.conv2.weight, "conv2.weight.txt")
save_tensor(model.conv2.bias,   "conv2.bias.txt")
save_tensor(model.conv3.weight, "conv3.weight.txt")
save_tensor(model.conv3.bias,   "conv3.bias.txt")
save_tensor(model.fc.weight,    "fc.weight.txt")
save_tensor(model.fc.bias,      "fc.bias.txt")

print("Weights exported again.")
