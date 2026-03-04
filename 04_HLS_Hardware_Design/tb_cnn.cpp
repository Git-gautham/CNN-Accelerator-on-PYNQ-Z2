#include <iostream>
#include <fstream>
#include "cnn_top.h"

int main() {

    std::ifstream file("cifar_test_100.txt");

    if(!file.is_open()){
        std::cout << "Error opening file!" << std::endl;
        return 1;
    }

    int correct = 0;
    int total = 100;

    for(int img = 0; img < total; img++) {

        data_t input[3072];
        data_t output[10];
        std::cout << "Reading image index: " << img << std::endl;
        std::cout << std::flush;
        int label;
        file >> label;

        // Load image
        for(int i = 0; i < 3072; i++){
            float temp;
            file >> temp;

            temp = (temp - 0.5f) / 0.5f;
            input[i] = temp;
        }
        

        // Run CNN
        cnn_top(input, output);

        // Argmax
        int pred = 0;
        data_t max_val = output[0];

        for(int i = 1; i < 10; i++){
            if(output[i] > max_val){
                max_val = output[i];
                pred = i;
            }
        }

        if(pred == label)
            correct++;

        std::cout << "Image " << img 
                  << ": True=" << label 
                  << " Pred=" << pred << std::endl;
    }

    float acc = (float)correct / total * 100.0;

    std::cout << "==============================" << std::endl;
    std::cout << "CSIM Accuracy: " << acc << " %" << std::endl;
    std::cout << "==============================" << std::endl;

    file.close();
    return 0;
}