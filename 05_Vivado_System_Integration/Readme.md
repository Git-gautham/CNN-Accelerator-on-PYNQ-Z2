This README is specifically for the **05_Vivado_System_Integration** directory. This stage is where you move from individual components to a complete "System-on-Chip" (SoC) by connecting your custom HLS accelerator to the Zynq-7000 Processing System (PS).

---

# 05. Vivado System Integration

This directory contains the Vivado project files and hardware descriptions required to integrate the CNN accelerator IP into a complete hardware system for the PYNQ-Z2 board.

## 📌 Overview

In this phase, we use the **Vivado IP Integrator** to build a block design that bridges the gap between the ARM processor (Software) and the FPGA logic (Hardware). The primary goals are:

1. **Block Design:** Connecting the CNN IP to the Zynq PS.
2. **Data Movement:** Setting up AXI Direct Memory Access (DMA) to stream data from DDR memory to the accelerator.
3. **Address Mapping:** Assigning memory addresses to the IP registers so they can be controlled via Python.
4. **Bitstream Generation:** Compiling the entire design into a `.bit` file and a `.hwh` (hardware handoff) file.

## 📂 Directory Structure

* `vivado_project/`: The Xilinx Vivado project files (`.xpr`).
* `src/`:
* `constraints.xdc`: Pin mapping and timing constraints (if physical I/O like LEDs are used).
* `block_design.tcl`: A script to automatically recreate the entire block design.


* `bitstream/`:
* `cnn_accelerator.bit`: The binary file used to program the FPGA.
* `cnn_accelerator.hwh`: The hardware handoff file required by the PYNQ library to identify the IP.



## 🏗 System Architecture

The hardware design typically consists of the following key components:

* **Zynq7 Processing System:** The "heart" of the system (ARM Cortex-A9).
* **CNN Accelerator IP:** The custom IP generated in folder `04_HLS_Hardware_Design`.
* **AXI DMA:** Facilitates high-speed data transfer between the DDR3 memory and the CNN IP.
* **AXI Interconnect/SmartConnect:** Routes data between the PS and various PL peripherals.
* **Processor System Reset:** Handles the synchronization of reset signals across clock domains.

## 🚀 Workflow

### 1. Recreating the Project

If you are using the provided scripts, you can rebuild the project in Vivado by running:

```tcl
# In the Vivado Tcl Console:
source ./05_Vivado_System_Integration/src/block_design.tcl

```

### 2. Integration Steps (Manual)

1. **Create a New Project** targeting the PYNQ-Z2 board.
2. **Add the IP Repository:** Point Vivado to the folder where you exported the HLS IP.
3. **Create Block Design:** Add the Zynq PS, the CNN IP, and the AXI DMA.
4. **Run Connection Automation:** Let Vivado handle the standard AXI and clock connections.
5. **Validate Design:** Ensure there are no critical warnings or design rule violations.

### 3. Generate Bitstream

Once the design is validated:

1. Click **Generate Bitstream**.
2. Export the hardware files. For PYNQ, you specifically need the `.bit` and the `.hwh` files. Rename them to have the same prefix (e.g., `cnn.bit` and `cnn.hwh`).

## 📊 Resource Utilization

After implementation, check the **Utilization Report**. The design must fit within the Zynq-7020 constraints:

* **LUTs:** ~53,200
* **DSPs:** 220
* **BRAM:** 4.9 Mb

---

**Next Step:** Move to `06_Jupyter_Notebook_Deployment` to upload your bitstream and run the accelerator using Python.
