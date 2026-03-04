This README is designed for the **04_HLS_Hardware_Design** directory. This is the "engine room" of the project, where C/C++ code is transformed into high-performance RTL (Register Transfer Level) hardware using Xilinx Vitis HLS or Vivado HLS.

---

# 04. HLS Hardware Design

This directory contains the High-Level Synthesis (HLS) source code for the CNN accelerator. Here, the architectural definitions of the convolutional layers, pooling, and activations are optimized for parallel execution on the PYNQ-Z2's FPGA fabric.

## 📌 Overview

Unlike standard software, the HLS code is written to be synthesized into hardware circuits. This design focuses on:

1. **Parallelism:** Using HLS pragmas (`#pragma HLS UNROLL`, `#pragma HLS PIPELINE`) to process multiple pixels or filters simultaneously.
2. **Memory Optimization:** Using line buffers and window buffers to minimize external memory access (DDR) and maximize on-chip BRAM usage.
3. **AXI Interfacing:** Implementing `m_axi` interfaces for high-bandwidth data movement (DMA) and `s_axilite` for control signals.

## 📂 Directory Structure

* `src/`: Core HLS C++ source files.
* `cnn_accelerator.cpp`: Main top-level function for the hardware IP.
* `layers.cpp`: Implementation of Conv2D, MaxPooling, and ReLU layers.
* `params.h`: Configuration for layer sizes, bit-widths, and fixed-point types.


* `tb/`: Testbench files.
* `cnn_tb.cpp`: C-simulation testbench to verify logic before synthesis.


* `scripts/`:
* `run_hls.tcl`: Tcl script to automate the synthesis and IP export process.



## ⚡ Optimization Techniques

The following HLS optimizations are employed to achieve high throughput:

* **Pipelining:** Enables the hardware to start processing a new input before the previous one has finished.
* **Loop Unrolling:** Replicates hardware logic to execute multiple loop iterations in a single clock cycle.
* **Array Partitioning:** Breaks down large arrays into smaller registers or BRAM blocks to allow simultaneous multi-port access.
* **Fixed-Point Arithmetic:** Uses `ap_fixed` types to reduce resource consumption compared to standard floating-point units.

## 🚀 Workflow

### 1. C-Simulation

Verify the functional correctness of the C++ code against the software baseline.

```bash
# In Vivado/Vitis HLS:
# Run C-Simulation to ensure output matches 03_ARM_Software_Baseline.

```

### 2. Synthesis & Co-Simulation

Synthesize the code into RTL and run C/RTL Co-simulation to verify the hardware timing and latency.

* Check the **Synthesis Report** for:
* **Latency:** Cycles required per inference.
* **Utilization:** Number of BRAMs, DSPs, LUTs, and FFs used (must fit PYNQ-Z2/Z7020 limits).



### 3. Export IP

Export the design as a Vivado IP Catalog block (`.zip`).

* This IP will be imported into the Vivado Block Design in the next stage of the project.

## 🛠 Prerequisites

* **Xilinx Vitis HLS** or **Vivado HLS** (v2019.1 or later recommended).
* Knowledge of HLS pragmas and FPGA resource constraints.

---

**Next Step:** Once the IP is exported, move to `05_Vivado_Hardware_Integration` to connect the accelerator to the Zynq Processing System.
