#include "cnn_top.h"
#include "cnn_weights_hls.h"
#include <ap_fixed.h>
#include <hls_math.h>

void cnn_top(
    volatile data_t *input,   // DDR pointer
    volatile data_t *output   // DDR pointer
) {

#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem \
                         depth=3072 max_read_burst_length=64  \
                         num_read_outstanding=16

#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem \
                         depth=10   max_write_burst_length=64 \
                         num_write_outstanding=16

#pragma HLS INTERFACE s_axilite port=input  bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control


    // ===============================
    // Local Buffers
    // ===============================

    data_t in_buf[IN_CH][IN_H][IN_W];

#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=2 dim=1

    data_t conv1_out[CONV1_CH][16][16];
    data_t conv2_out[CONV2_CH][8][8];
    data_t conv3_out[CONV3_CH][4][4];

#pragma HLS ARRAY_PARTITION variable=conv1_out cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=conv2_out cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=conv3_out cyclic factor=2 dim=1


    // ===============================
    // Load input from DDR
    // ===============================

    int idx = 0;
LOAD_INPUT:
    for(int c=0;c<IN_CH;c++){
        for(int i=0;i<IN_H;i++){
            for(int j=0;j<IN_W;j++){
#pragma HLS PIPELINE II=1
                in_buf[c][i][j] = input[idx++];
            }
        }
    }


    // ===============================
    // Layer 1
    // ===============================

L1_OC:
    for(int oc=0; oc<CONV1_CH; oc++){
        for(int i=0;i<16;i++){
            for(int j=0;j<16;j++){

                data_t sum = conv1_b[oc];

                for(int ic=0; ic<IN_CH; ic++){
                    for(int ki=0; ki<3; ki++){
                        for(int kj=0; kj<3; kj++){

#pragma HLS PIPELINE II=1

                            int in_i = i*2 + ki - 1;
                            int in_j = j*2 + kj - 1;

                            data_t val = 0;

                            if(in_i>=0 && in_i<32 && in_j>=0 && in_j<32)
                                val = in_buf[ic][in_i][in_j];

                            int widx = oc*IN_CH*9 + ic*9 + ki*3 + kj;

                            sum += val * conv1_w[widx];
                        }
                    }
                }

                conv1_out[oc][i][j] = hls::max(sum,(data_t)0);
            }
        }
    }


    // ===============================
    // Layer 2
    // ===============================

L2_OC:
    for(int oc=0; oc<CONV2_CH; oc++){
        for(int i=0;i<8;i++){
            for(int j=0;j<8;j++){

                data_t sum = conv2_b[oc];

                for(int ic=0; ic<CONV1_CH; ic++){
                    for(int ki=0; ki<3; ki++){
                        for(int kj=0; kj<3; kj++){

#pragma HLS PIPELINE II=1

                            int in_i = i*2 + ki - 1;
                            int in_j = j*2 + kj - 1;

                            data_t val = 0;

                            if(in_i>=0 && in_i<16 && in_j>=0 && in_j<16)
                                val = conv1_out[ic][in_i][in_j];

                            int widx = oc*CONV1_CH*9 + ic*9 + ki*3 + kj;

                            sum += val * conv2_w[widx];
                        }
                    }
                }

                conv2_out[oc][i][j] = hls::max(sum,(data_t)0);
            }
        }
    }


    // ===============================
    // Layer 3
    // ===============================

L3_OC:
    for(int oc=0; oc<CONV3_CH; oc++){
        for(int i=0;i<4;i++){
            for(int j=0;j<4;j++){

                data_t sum = conv3_b[oc];

                for(int ic=0; ic<CONV2_CH; ic++){
                    for(int ki=0; ki<3; ki++){
                        for(int kj=0; kj<3; kj++){

#pragma HLS PIPELINE II=1

                            int in_i = i*2 + ki - 1;
                            int in_j = j*2 + kj - 1;

                            data_t val = 0;

                            if(in_i>=0 && in_i<8 && in_j>=0 && in_j<8)
                                val = conv2_out[ic][in_i][in_j];

                            int widx = oc*CONV2_CH*9 + ic*9 + ki*3 + kj;

                            sum += val * conv3_w[widx];
                        }
                    }
                }

                conv3_out[oc][i][j] = hls::max(sum,(data_t)0);
            }
        }
    }


    // ===============================
    // Fully Connected
    // ===============================

FC_OUT;
    for(int o=0; o<FC_OUT; o++){

        data_t sum = fc_b[o];

        for(int ic=0; ic<CONV3_CH; ic++){
            for(int i=0;i<4;i++){
                for(int j=0;j<4;j++){

#pragma HLS PIPELINE II=1

                    int widx = o*CONV3_CH*16 + ic*16 + i*4 + j;

                    sum += conv3_out[ic][i][j] * fc_w[widx];
                }
            }
        }

        output[o] = sum;
    }
}