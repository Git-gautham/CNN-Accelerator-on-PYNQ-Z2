

# Real-Time Object Detection on Xilinx Zynq-7000 (PYNQ-Z2)

**A Hardware-Accelerated CNN Implementation for the Arm Bharat AI-SoC Student Challenge.**

## 🚀 Project Overview

This project demonstrates a full-stack Hardware/Software co-design to accelerate CNN inference. We implement a 3-layer Convolutional Neural Network on the **TUL PYNQ-Z2** board, offloading compute-heavy layers from the Arm CPU to the FPGA fabric.

### Key Achievements:

* **Heterogeneous Computing:** Partitioned workload between Arm Cortex-A9 (Control) and FPGA Logic (Acceleration).
* **Optimization:** Used Vitis HLS pragmas (Pipelining, Unrolling) and 16-bit quantization.
* **Performance:** Achieved a significant speedup compared to a standalone C++ sequential implementation.

---

## 🛠️ System Architecture

The system is divided into two main domains:

1. **Processing System (PS):** - Image pre-processing and normalization in Python/C++.
* Control logic and DMA management.


2. **Programmable Logic (PL):** - Custom CNN IP core synthesized via Vitis HLS.
* AXI-Stream interfaces for high-speed data movement.



---

## 📁 Repository Structure

* `/01_Training`: PyTorch training and 16-bit weight quantization scripts.
* `/02_Software_Baseline`: Sequential C++ implementation for ARM CPU performance measurement.
* `/03_HLS_IP`: Source code for the hardware accelerator (`cnn_accelerator.cpp`).
* `/04_Vivado_Project`: Bitstream and hardware handover files (`.bit`, `.hwh`).
* `/05_PYNQ_Deployment`: Jupyter Notebook for real-time inference on the PYNQ-Z2.

---

## 📊 Performance Comparison (CIFAR-10)

Results based on a 100-image test batch:
<img width="735" height="303" alt="image" src="https://github.com/user-attachments/assets/18f38e4e-973f-4153-a96d-21d46bc9fa6f" />


## 🛠️ How to Run

1. **Hardware Setup:** Connect your PYNQ-Z2 to your network and power it on.
2. **Upload Overlay:** Copy the `.bit` and `.hwh` files from `/04_Vivado_Project` to your PYNQ board.
3. **Execute Notebook:** Open the notebook in `/05_PYNQ_Deployment` via the Jupyter interface.
4. **Run Inference:** Follow the cells to load the overlay, transfer weights, and perform real-time classification.

---

## 👥 Contributors

* **Gaurynandana J**
* **Gautham P Nair**
* **Lakshmi Nair**
* *Govt. Model Engineering College, Thrikkakara*
* Mentor : Dr Jagadeesh Kumar P
---
