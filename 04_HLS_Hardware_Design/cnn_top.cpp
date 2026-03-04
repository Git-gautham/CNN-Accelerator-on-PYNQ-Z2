#include "cnn_top.h"
#include "cnn_weights_hls.h"
#include <hls_math.h>

void cnn_top(
    volatile data_t *input,
    volatile data_t *output
) {

#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem depth=INPUT_SIZE
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem depth=OUTPUT_SIZE

#pragma HLS INTERFACE s_axilite port=input  bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control


// ======================================================
// Buffers
// ======================================================

data_t in_buf[IN_CH][IN_H][IN_W];

data_t conv1_out[CONV1_CH][CONV1_H][CONV1_W];
data_t pool1_out[CONV1_CH][POOL1_H][POOL1_W];

data_t conv2_out[CONV2_CH][CONV2_H][CONV2_W];
data_t pool2_out[CONV2_CH][POOL2_H][POOL2_W];

data_t conv3_out[CONV3_CH][CONV3_H][CONV3_W];
data_t pool3_out[CONV3_CH][POOL3_H][POOL3_W];

#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=2 dim=1


// ======================================================
// Load Input
// ======================================================

int idx = 0;

LOAD_INPUT:
for(int c=0;c<IN_CH;c++)
    for(int i=0;i<IN_H;i++)
        for(int j=0;j<IN_W;j++){
#pragma HLS PIPELINE II=1
            in_buf[c][i][j] = input[idx++];
        }


// ======================================================
// LAYER 1  (Stride 1, Pad 1)
// ======================================================

L1_OC:
for(int oc=0; oc<CONV1_CH; oc++)
for(int i=0;i<CONV1_H;i++)
for(int j=0;j<CONV1_W;j++){

    data_t sum = conv1_b[oc];

    for(int ic=0; ic<IN_CH; ic++)
    for(int ki=0; ki<3; ki++)
    for(int kj=0; kj<3; kj++){

#pragma HLS PIPELINE II=1

        int in_i = i + ki - 1;
        int in_j = j + kj - 1;

        if(in_i>=0 && in_i<IN_H &&
           in_j>=0 && in_j<IN_W){

            int widx =
                oc*IN_CH*9 +
                ic*9 + ki*3 + kj;

            sum += in_buf[ic][in_i][in_j]
                   * conv1_w[widx];
        }
    }

    conv1_out[oc][i][j] =
        hls::max(sum,(data_t)0);
}


// ======================================================
// MAXPOOL 1 (2x2)
// ======================================================

POOL1:
for(int c=0;c<CONV1_CH;c++)
for(int i=0;i<POOL1_H;i++)
for(int j=0;j<POOL1_W;j++){

    data_t m = conv1_out[c][i*2][j*2];

    for(int ki=0;ki<2;ki++)
    for(int kj=0;kj<2;kj++){

        data_t val =
            conv1_out[c][i*2+ki][j*2+kj];

        if(val > m) m = val;
    }

    pool1_out[c][i][j] = m;
}


// ======================================================
// LAYER 2
// ======================================================

L2_OC:
for(int oc=0; oc<CONV2_CH; oc++)
for(int i=0;i<CONV2_H;i++)
for(int j=0;j<CONV2_W;j++){

    data_t sum = conv2_b[oc];

    for(int ic=0; ic<CONV1_CH; ic++)
    for(int ki=0; ki<3; ki++)
    for(int kj=0; kj<3; kj++){

#pragma HLS PIPELINE II=1

        int in_i = i + ki - 1;
        int in_j = j + kj - 1;

        if(in_i>=0 && in_i<POOL1_H &&
           in_j>=0 && in_j<POOL1_W){

            int widx =
                oc*CONV1_CH*9 +
                ic*9 + ki*3 + kj;

            sum += pool1_out[ic][in_i][in_j]
                   * conv2_w[widx];
        }
    }

    conv2_out[oc][i][j] =
        hls::max(sum,(data_t)0);
}


// ======================================================
// MAXPOOL 2
// ======================================================

POOL2:
for(int c=0;c<CONV2_CH;c++)
for(int i=0;i<POOL2_H;i++)
for(int j=0;j<POOL2_W;j++){

    data_t m = conv2_out[c][i*2][j*2];

    for(int ki=0;ki<2;ki++)
    for(int kj=0;kj<2;kj++){

        data_t val =
            conv2_out[c][i*2+ki][j*2+kj];

        if(val > m) m = val;
    }

    pool2_out[c][i][j] = m;
}


// ======================================================
// LAYER 3
// ======================================================

L3_OC:
for(int oc=0; oc<CONV3_CH; oc++)
for(int i=0;i<CONV3_H;i++)
for(int j=0;j<CONV3_W;j++){

    data_t sum = conv3_b[oc];

    for(int ic=0; ic<CONV2_CH; ic++)
    for(int ki=0; ki<3; ki++)
    for(int kj=0; kj<3; kj++){

#pragma HLS PIPELINE II=1

        int in_i = i + ki - 1;
        int in_j = j + kj - 1;

        if(in_i>=0 && in_i<POOL2_H &&
           in_j>=0 && in_j<POOL2_W){

            int widx =
                oc*CONV2_CH*9 +
                ic*9 + ki*3 + kj;

            sum += pool2_out[ic][in_i][in_j]
                   * conv3_w[widx];
        }
    }

    conv3_out[oc][i][j] =
        hls::max(sum,(data_t)0);
}


// ======================================================
// MAXPOOL 3
// ======================================================

POOL3:
for(int c=0;c<CONV3_CH;c++)
for(int i=0;i<POOL3_H;i++)
for(int j=0;j<POOL3_W;j++){

    data_t m = conv3_out[c][i*2][j*2];

    for(int ki=0;ki<2;ki++)
    for(int kj=0;kj<2;kj++){

        data_t val =
            conv3_out[c][i*2+ki][j*2+kj];

        if(val > m) m = val;
    }

    pool3_out[c][i][j] = m;
}


// ======================================================
// FULLY CONNECTED
// ======================================================

FC_LAYER:
for(int o=0;o<FC_OUT;o++){

    data_t sum = fc_b[o];

    for(int ic=0; ic<CONV3_CH; ic++)
    for(int i=0;i<POOL3_H;i++)
    for(int j=0;j<POOL3_W;j++){

#pragma HLS PIPELINE II=1

        int widx =
            o*FC_IN +
            ic*(POOL3_H*POOL3_W) +
            i*POOL3_W + j;

        sum += pool3_out[ic][i][j]
               * fc_w[widx];
    }

    output[o] = sum;
}

}