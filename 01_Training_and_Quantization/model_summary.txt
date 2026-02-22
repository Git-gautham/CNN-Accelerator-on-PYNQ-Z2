
## Model Summary

### **Architecture Overview:**
The model is a lightweight Convolutional Neural Network (CNN) designed for CIFAR-10 image classification (32×32 RGB images).

### **Layers:**

| Layer | Details |
|-------|---------|
| **Conv Layer 1** | 3 → 16 filters, 3×3 kernel, padding=1 |
| **ReLU** | Activation function |
| **MaxPool 1** | 2×2 pooling |
| **Conv Layer 2** | 16 → 32 filters, 3×3 kernel, padding=1 |
| **ReLU** | Activation function |
| **MaxPool 2** | 2×2 pooling |
| **Conv Layer 3** | 32 → 64 filters, 3×3 kernel, padding=1 |
| **ReLU** | Activation function |
| **MaxPool 3** | 2×2 pooling |
| **Fully Connected** | 64 × 4 × 4 = 1,024 → 10 (output classes) |

### **Key Specifications:**
- **Input Shape:** 3 × 32 × 32 (RGB CIFAR-10 images)
- **Output Shape:** 10 classes (CIFAR-10 categories)
- **Activation:** ReLU
- **Pooling:** Max pooling (2×2) after each convolutional layer
- **Spatial Reduction:** 32×32 → 4×4 after 3 pooling operations

### **Training Configuration:**
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 10
- **Batch Size:** 64
- **Data:** CIFAR-10 (normalized to [-1, 1])

This is a simple but effective architecture for baseline CIFAR-10 classification tasks, suitable for hardware acceleration projects like the PYNQ-Z2 platform.
