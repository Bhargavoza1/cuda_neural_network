#include <iostream>

#include <stdexcept>
 
#include"Tensor.h"
#include "tensor_oprations.cuh"

using namespace Hex;
using namespace std;

template <class T>
void printtensor(const Tensor<T>& tensor1) {
    tensor1.print();
}


int main() {
     
    // Create tensors with shape [3, 2048, 1080]
    std::unique_ptr<Tensor<float>> tensorA(new Tensor<float>({300, 3, 2048, 1080 }));
    std::unique_ptr<Tensor<int>> tensorB(new Tensor<int>({ 300,3, 2048, 1080 }));

    // Initialize tensors on GPU
    initTensorOnGPU(*tensorA , 0.2);
    initTensorOnGPU(*tensorB , 0.0);
    std::cout << "tensor init done" << std::endl;
    // Perform element-wise addition for 3D tensors
    auto tensorC = Hex::addTensor(*tensorA, *tensorB);


    // Print the result tensor
   // std::cout << "Tensor C (A + B):" << std::endl;
   // printtensor(*tensorC);

    return 0;
}
