This README is specifically tailored for the **01_Training_and_Quantization** directory of your project. It assumes a typical workflow for FPGA-based CNN acceleration (like the PYNQ-Z2), where models are trained in high-level frameworks and then quantized to fixed-point integers to save hardware resources.

---

# 01. Training and Quantization

This directory contains the scripts and notebooks required to design, train, and quantize the Convolutional Neural Network (CNN) before deploying it to the PYNQ-Z2 hardware.

## 📌 Overview

FPGAs perform significantly better with integer arithmetic (Fixed-point) than with floating-point operations. This stage of the pipeline focuses on:

1. **Model Definition:** Defining a CNN architecture suitable for hardware acceleration.
2. **Training:** Training the model on a dataset (e.g., MNIST, CIFAR-10) using standard deep learning frameworks.
3. **Quantization:** Converting weights and biases from `float32` to fixed-point integers (e.g., `int8` or `int16`) to match the hardware accelerator's bit-width.
4. **Weight Extraction:** Exporting the trained parameters into headers or binary files for use in Vivado HLS or the C++ driver.

## 📂 Directory Structure

* `training_script.ipynb / .py`: The main script for training the model.
* `quantization_logic.py`: Functions to scale and clip weights for fixed-point representation.
* `models/`: Saved model files (e.g., `.h5`, `.pth`, or `.onnx`).
* `exported_data/`: The final quantized weights and biases (usually in `.h` or `.txt` format) ready for the hardware.

## 🛠 Prerequisites

Ensure you have the following Python environment set up:

```bash
pip install tensorflow torch torchvision numpy matplotlib

```

*(Adjust libraries based on whether you used PyTorch or TensorFlow)*

## 🚀 How to Use

### 1. Training the Model

Run the training script to generate a high-precision floating-point model.

```bash
python training_script.py

```

### 2. Quantization Process

The quantization step maps the floating-point range (e.g., -1.0 to 1.0) to an integer range (e.g., -128 to 127 for 8-bit).

* **Bit-width:** Default is set to 8-bit/16-bit (match this with your HLS configuration).
* **Scaling Factor:** The script calculates the optimal fractional bits to minimize accuracy loss.

### 3. Exporting for Hardware

Once quantized, the script will generate files (often `weights.h` or `biases.h`). These files should be moved to your **HLS (High-Level Synthesis)** directory or loaded directly via the PYNQ Python overlay.

## 📊 Results

After quantization, you should verify the "Quantized Accuracy" vs. "Floating-point Accuracy" to ensure the bit-width reduction hasn't significantly degraded the model's performance.

| Model Version | Accuracy (%) |
| --- | --- |
| Floating Point (32-bit) | XX.X% |
| Quantized (Fixed-point) | XX.X% |

---

**Next Step:** Proceed to the `02_HLS_Hardware_Design` folder to implement the architecture in RTL/HLS.
