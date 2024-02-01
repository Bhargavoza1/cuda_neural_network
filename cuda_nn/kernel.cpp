#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include"Tensor.h"
#include "tensor_oprations.cpp"
using namespace Hex;


int main() {
    // Create tensors with shapes [2, 3, 4] and [2, 3, 4]
    Tensor<double> tensorA({ 2, 3, 4 });
    Tensor<int> tensorB({ 2, 3, 4 }); 
    // Initialize input data for 3D tensors
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                float val = static_cast<float>(i * 3 * 4 + j * 4 + k);
                tensorA.set({ i, j, k }, val * 0.2);
                tensorB.set({ i, j, k }, 2 * val);
            }
        }
    }

    // Perform element-wise addition for 3D tensors
    auto tensorC = Hex::addTensor(tensorA , tensorB);

    std::cout << "Tensor C (A + B):" << std::endl;
    tensorC->print();

    return 0;
}
