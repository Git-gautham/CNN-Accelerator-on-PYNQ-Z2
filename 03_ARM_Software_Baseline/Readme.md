This README is specifically for the **03_ARM_Software_Baseline** directory. In an FPGA acceleration project, the software baseline is the "ground truth" performance. It allows you to measure how much faster your hardware accelerator (FPGA) is compared to the onboard ARM processor (CPU) of the PYNQ-Z2.

---

# 03. ARM Software Baseline

This directory contains the Python-based implementation of the CNN layers designed to run entirely on the **ARM Cortex-A9 processor** of the PYNQ-Z2. This serves as the reference point for evaluating the performance gains achieved by the FPGA hardware accelerator.

## 📌 Overview

Before deploying to the FPGA (Programmable Logic), we implement the model in software (Processing System). This allows us to:

1. **Verify Logic:** Ensure the mathematical implementation of convolution, pooling, and activation functions is correct.
2. **Benchmark Performance:** Measure the execution time and power consumption of the model running on a standard CPU.
3. **Accuracy Check:** Establish the baseline accuracy for the quantized weights when processed through standard software libraries (NumPy).

## 📂 Directory Structure

* `software_inference.ipynb / .py`: The primary script to load weights and run inference on the ARM CPU.
* `cnn_layers_baseline.py`: Custom Python/NumPy implementations of the CNN operations (Conv2D, ReLU, MaxPool, FC) without hardware acceleration.
* `benchmarking_results.json`: (Optional) Logs of execution times for different batch sizes.

## 🛠 Prerequisites

These scripts are intended to run on the PYNQ-Z2 board via the Jupyter Notebook interface or via SSH.

```bash
# Standard PYNQ libraries (usually pre-installed)
pip install numpy matplotlib time

```

## 🚀 How to Use

### 1. Run the Baseline Inference

Load the quantized weights from the `02_Weights_and_Datasets` folder and execute the software-only inference:

```python
from cnn_layers_baseline import forward_pass

# Load data and weights
input_data = ... 
weights = ...

# Measure execution time
import time
start = time.time()
output = forward_pass(input_data, weights)
end = time.time()

print(f"Software Inference Time: {end - start:.4f} seconds")

```

### 2. Comparison Metrics

This baseline is used to calculate the **Speedup Factor**:


$$\text{Speedup} = \frac{\text{Execution Time (ARM Software)}}{\text{Execution Time (FPGA Hardware)}}$$

## 📊 Expected Performance

The ARM Cortex-A9 is a general-purpose processor. For a typical CNN (like LeNet or a custom small-scale network), you can expect:

* **Latency:** High (milliseconds to seconds per image).
* **Throughput:** Low (limited by sequential processing of nested loops in convolution).
* **Accuracy:** Should match the "Quantized Accuracy" from step 01 exactly.

---

**Next Step:** Use these results to compare against the hardware implementation in `04_Hardware_Accelerator_Implementation`.
