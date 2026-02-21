import torchvision
import torchvision.transforms as transforms
import numpy as np

transform = transforms.ToTensor()

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

with open("cifar_test.txt", "w") as f:
    for img, label in testset:
        img = img.numpy().flatten()
        f.write(str(label) + "\n")
        for v in img:
            f.write(str(v) + " ")
        f.write("\n")

print("Full CIFAR-10 test set exported.")
correct = 0
total = 0

with torch.no_grad():
    for img, label in testset:
        img = img.unsqueeze(0).to(device)
        out = model(img)
        pred = torch.argmax(out, 1).item()
        if pred == label:
            correct += 1
        total += 1

print("Python accuracy:", correct/total*100)
