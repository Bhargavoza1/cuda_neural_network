#include <iostream>

#include <stdexcept>
 
#include"Tensor.h"
#include "tensor_oprations.cuh"
#include "linear.h"
using namespace Hex;
using namespace std;

template <class T>
void printtensor(const Tensor<T>& tensor1) {
    tensor1.print();
}


int main() {
     
    // memory get overloaded on {330, 3, 2048, 1080 }
    //std::unique_ptr<Tensor<int>> tensorA(new Tensor<int>({330, 3, 2048, 1080 }));
    //std::unique_ptr<Tensor<int>> tensorB(new Tensor<int>({ 330,3, 2048, 1080 }));
    std::unique_ptr<Tensor<float>> tensorA(new Tensor<float>({30 , 3, 2048, 1080 }));
    std::unique_ptr<Tensor<int>> tensorB(new Tensor<int>({ 30 ,3, 2048, 1080 }));
    linear<float> linearLayer(10,5    );
    
    // Initialize tensors on GPU
    initTensorOnGPU(*tensorA , 0.0);
    initTensorOnGPU(*tensorB , 0.0);
    std::cout << "tensor init done" << std::endl;
    // Perform element-wise addition for 3D tensors
    auto tensorC = Hex::addTensor(*tensorA, *tensorB);

  
    // Print the result tensor
     std::cout << "Tensor C (A + B):" << std::endl;
     printtensor(linearLayer.forward(*tensorA));

    return 0;
}
