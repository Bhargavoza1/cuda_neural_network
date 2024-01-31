// main.cpp

#include <iostream>
#include <random>
#include "tensor.h"

int main() {
    // Create a 3x4x5x6 tensor with random double values
    Hex::Tensor<double> gpuTensor({3, 4, 5, 6});

    // Generate random numbers for the tensor data (modify the distribution)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 5.0);
    int value = 0;
    
    // Calculate the shape once and store it in a variable
    const auto shape = gpuTensor.getShape();
 
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            for (size_t k = 0; k < shape[2]; ++k) {
                for (size_t l = 0; l < shape[3]; ++l) {
                    gpuTensor({i, j, k, l}) = static_cast<double>(value++);
                }
            }
        }
    }

   
    // Display shape and data
    std::cout << "Tensor Shape: ";
    for (size_t dim : shape) {
        std::cout << dim << "x";
    }
    std::cout << "\b \n";

    std::cout << "Tensor Data:\n";
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            std::cout << "\n" << "tensor number:" << i << "," << j << "\n";
            for (size_t k = 0; k < shape[2]; ++k) {
                for (size_t l = 0; l < shape[3]; ++l) {
                    std::cout << gpuTensor({i, j, k, l}) << " ";
                }
                std::cout << "\n";
            }
        }
    }

    return 0;
}
