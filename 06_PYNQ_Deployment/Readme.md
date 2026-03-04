This README is designed for the **06_PYNQ_Deployment** directory. This is the final stage of the project, where the hardware and software converge on the PYNQ-Z2 board to perform real-time, hardware-accelerated inference.

---

# 06. PYNQ Deployment

This directory contains the high-level Python code and Jupyter Notebooks required to interact with the FPGA hardware. Using the PYNQ framework, we load the custom bitstream, manage data transfers via DMA, and execute the CNN acceleration.

## 📌 Overview

The deployment phase focuses on the "User Experience" of the accelerator. It abstracts the complex hardware signals into simple Python commands. Key tasks include:

1. **Overlay Loading:** Programming the FPGA with the `.bit` file generated in the Vivado stage.
2. **Memory Management:** Allocating contiguous memory (CMA) buffers for input images and output results.
3. **DMA Control:** Streaming data from the ARM processor's RAM to the FPGA's processing engine.
4. **Performance Profiling:** Measuring the exact hardware execution time and comparing it against the software baseline.

## 📂 Directory Structure

* `cnn_deployment.ipynb`: The primary Jupyter Notebook for running the end-to-end acceleration demo.
* `overlay/`:
* `cnn_accelerator.bit`: The FPGA binary configuration file.
* `cnn_accelerator.hwh`: The hardware handoff file (defines IP register maps for PYNQ).


* `utils/`:
* `plotting.py`: Helper functions to visualize input images and classification results.
* `hardware_drivers.py`: Custom Python wrapper classes to simplify interactions with the HLS IP.



## 🛠 Prerequisites

These files are intended to be uploaded directly to your PYNQ-Z2 board (typically at `http://192.168.2.99`).

* **Board:** PYNQ-Z2
* **Image:** PYNQ v2.5 or later
* **Libraries:** `pynq`, `numpy`, `matplotlib`

## 🚀 How to Run

### 1. Load the Hardware Overlay

In the Jupyter Notebook, initialize the overlay to program the FPGA:

```python
from pynq import Overlay
overlay = Overlay("overlay/cnn_accelerator.bit")

# Access the DMA and the Accelerator IP
dma = overlay.axi_dma_0
cnn_ip = overlay.cnn_accelerator_0

```

### 2. Prepare Buffers

Use PYNQ's `xlnk` or `allocate` to create memory-mapped buffers that the FPGA can access directly:

```python
from pynq import allocate
input_buffer = allocate(shape=(1, 28, 28), dtype='i1') # For MNIST
output_buffer = allocate(shape=(10,), dtype='i1')

```

### 3. Execute Acceleration

Send the data, start the hardware core, and wait for the results:

```python
# Copy data to buffer and sync with device
input_buffer[:] = test_image
input_buffer.flush()

# Trigger DMA and IP
dma.sendchannel.transfer(input_buffer)
cnn_ip.write(0x00, 0x1) # Start signal
dma.recvchannel.transfer(output_buffer)
dma.recvchannel.wait()

print(f"Predicted Class: {np.argmax(output_buffer)}")

```

## 📈 Analysis

The notebook includes sections to:

* **Visualize Predictions:** Display the input image alongside the top-3 predicted classes.
* **Timing Comparison:** A side-by-side bar chart showing **ARM Software Time** vs. **FPGA Hardware Time**.
* **Efficiency:** Calculation of throughput (Inferences Per Second).

---

**Congratulations!** You have completed the full pipeline from Training to FPGA Hardware Acceleration.
