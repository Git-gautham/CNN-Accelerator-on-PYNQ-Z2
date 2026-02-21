#ifndef CNN_TOP_H
#define CNN_TOP_H

#include <ap_fixed.h>

// ==============================
// Input dimensions
// ==============================

#define IN_CH 3
#define IN_H  32
#define IN_W  32

// ==============================
// Network dimensions
// ==============================

#define CONV1_CH 16
#define CONV2_CH 32
#define CONV3_CH 64

#define FC_OUT 10

// ==============================
// Data type
// ==============================

typedef ap_fixed<16,6> data_t;

// ==============================
// Derived sizes
// ==============================

#define INPUT_SIZE  (IN_CH * IN_H * IN_W)   // 3072
#define OUTPUT_SIZE (FC_OUT)                // 10

// ==============================
// Top function (AXI DDR pointers)
// ==============================

void cnn_top(
    volatile data_t *input,   // DDR memory pointer
    volatile data_t *output   // DDR memory pointer
);

#endif