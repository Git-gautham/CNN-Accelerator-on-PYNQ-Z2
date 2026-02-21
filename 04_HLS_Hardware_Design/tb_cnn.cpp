#include "cnn_top.h"
#include <iostream>

int main() {
    data_t input[3][32][32] = {0}; // Example input
    data_t output[10];

    // Fill input with test values (0~1 or from dataset)
    for(int c=0;c<3;c++)
        for(int i=0;i<32;i++)
            for(int j=0;j<32;j++)
                input[c][i][j] = (data_t)((c + i + j) % 256) / (data_t)255.0;

    cnn_top(input, output);

    std::cout << "CNN output:\n";
    for(int i=0;i<10;i++)
        std::cout << output[i] << " ";
    std::cout << std::endl;

    return 0;
}