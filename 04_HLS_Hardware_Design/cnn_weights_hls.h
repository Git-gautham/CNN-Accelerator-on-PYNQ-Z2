#ifndef CNN_WEIGHTS_HLS_H
#define CNN_WEIGHTS_HLS_H

#include <ap_fixed.h>

// Fixed-point type
typedef ap_fixed<16,6> data_t;

// Conv1
extern data_t conv1_w[432];
extern data_t conv1_b[16];

// Conv2
extern data_t conv2_w[4608];
extern data_t conv2_b[32];

// Conv3
extern data_t conv3_w[18432];
extern data_t conv3_b[64];

// Fully Connected
extern data_t fc_w[10240];
extern data_t fc_b[10];

#endif // CNN_WEIGHTS_HLS_H