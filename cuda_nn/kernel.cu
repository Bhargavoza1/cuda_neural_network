// main.cpp

#include <iostream>
#include <random>
#include "tensor.h"

int main() {
    // Create a 5x4x3 tensor with random double values
    Hex::Tensor<double> cpuTensor({ 5, 4, 3 });

    // Generate random numbers for the tensor data (modify the distribution)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 5.0);

    for (size_t i = 0; i < cpuTensor.getShape()[0]; ++i) {
        for (size_t j = 0; j < cpuTensor.getShape()[1]; ++j) {
            for (size_t k = 0; k < cpuTensor.getShape()[2]; ++k) {
                cpuTensor({ i, j, k }) = dis(gen);
            }
        }
    }

    // Display shape and data
    std::cout << "Tensor Shape: ";
    for (size_t dim : cpuTensor.getShape()) {
        std::cout << dim << "x";
    }
    std::cout << "\b \n";

    std::cout << "Tensor Data:\n";
    for (size_t i = 0; i < cpuTensor.getShape()[0]; ++i) {
        std::cout << "\n" << "tensor number:" << i << "\n";
        for (size_t j = 0; j < cpuTensor.getShape()[1]; ++j) {
            for (size_t k = 0; k < cpuTensor.getShape()[2]; ++k) {
                std::cout << cpuTensor({ i, j, k }) << " ";
            }
            std::cout << "\n";
        }
    }

    return 0;
}