
# 02. Weights and Datasets

This directory serves as the storage hub for the pre-processed data required to feed the CNN accelerator. It contains both the fixed-point weight parameters exported from the training phase and the datasets (or test images) used for inference on the PYNQ-Z2 board.

## 📌 Overview

Hardware acceleration requires data to be in a specific format (e.g., flattened arrays, memory-mapped binary files, or H-header files). This folder organizes these assets to ensure they are easily accessible by the PYNQ Python runtime or the C++ hardware driver.

## 📂 Directory Structure

* `/weights`: Contains the quantized model parameters (weights and biases).
* `weights.h`: C-header files containing constant arrays for integration into HLS/Vivado.
* `weights.bin`: Binary dumps for direct loading into DMA/memory.


* `/datasets`: Contains the input data for testing.
* `test_images/`: A subset of the dataset (e.g., MNIST digits) formatted for the accelerator.
* `labels.txt`: Ground truth labels corresponding to the test images.


* `process_data.py`: A helper utility to convert raw input images into the specific format required by your input buffers (e.g., resizing, grayscale conversion, normalization).

## 🛠 Preparation Requirements

Before running the deployment, ensure your data conforms to the hardware requirements:

1. **Fixed-point Precision:** Confirm that the weights in `/weights` match the bit-width (e.g., 8-bit signed) used in your FPGA HLS core.
2. **Memory Alignment:** If using DMA, ensure that the image data in `/datasets` is correctly padded or aligned to the burst size of the DMA engine.

## 🚀 Usage

### 1. Preparing Weights for Hardware

If your HLS project requires header files, use the provided script to convert binary/numpy weights into the C++ format:

```bash
python convert_weights_to_header.py --input weights.npy --output weights.h

```

### 2. Preparing Input Images

The PYNQ overlay expects data in a specific tensor shape. Use the preprocessing script to prepare a single image for inference:

```bash
# Example: Convert an image to a raw format compatible with the accelerator
python process_data.py --image input_digit.png --output input_buffer.bin

```

### 3. Loading Data via PYNQ

In your main deployment notebook, you will use these files as follows:

```python
# Example: Loading weights and data into the overlay
import numpy as np
from pynq import Overlay

# Load the weights from this directory
weights = np.fromfile("02_Weights_and_Datasets/weights/weights.bin", dtype=np.int8)

# Load the input image
image_data = np.fromfile("02_Weights_and_Datasets/datasets/test_images/sample_01.bin", dtype=np.int8)

```

## ⚠️ Data Integrity Note

* **Endianness:** Ensure the byte order of your binary files matches the CPU architecture of the Zynq-7000 (Little Endian).
* **Scaling:** If you observe incorrect outputs, verify that the scaling factors applied during the training phase (in `01_Training_and_Quantization`) have been correctly accounted for when preparing these binary files.

---

**Next Step:** Once your weights and input buffers are ready, proceed to the deployment phase to run inference on the PYNQ-Z2 board.
