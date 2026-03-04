#ifndef CNN_TOP_H
#define CNN_TOP_H

#include <ap_fixed.h>

// =====================================================
// Input dimensions
// =====================================================

#define IN_CH 3
#define IN_H  32
#define IN_W  32

// =====================================================
// Convolution Channels
// =====================================================

#define CONV1_CH 16
#define CONV2_CH 32
#define CONV3_CH 64

// =====================================================
// Spatial Dimensions (Stride 1 + 2x2 MaxPool)
// =====================================================

// After Conv1 (stride 1, pad 1)
#define CONV1_H 32
#define CONV1_W 32

// After Pool1
#define POOL1_H 16
#define POOL1_W 16

// After Conv2
#define CONV2_H 16
#define CONV2_W 16

// After Pool2
#define POOL2_H 8
#define POOL2_W 8

// After Conv3
#define CONV3_H 8
#define CONV3_W 8

// After Pool3
#define POOL3_H 4
#define POOL3_W 4

// =====================================================
// Fully Connected
// =====================================================

#define FC_OUT 10

// Number of FC input features
#define FC_IN  (CONV3_CH * POOL3_H * POOL3_W)   // 64*4*4 = 1024

// =====================================================
// Data Type (Quantized)
// =====================================================

typedef ap_fixed<16,6> data_t;

// =====================================================
// Derived sizes
// =====================================================

#define INPUT_SIZE  (IN_CH * IN_H * IN_W)   // 3072
#define OUTPUT_SIZE (FC_OUT)                // 10

// =====================================================
// Top Function Declaration (AXI DDR pointers)
// =====================================================

#ifdef __cplusplus
extern "C" {
#endif

void cnn_top(
    volatile data_t *input,
    volatile data_t *output
);

#ifdef __cplusplus
}
#endif

#endif