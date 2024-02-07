#include <iostream>

#include <stdexcept>

#include"Tensor.h"
#include "tensor_oprations.h"
#include "linear.h"
#include "ReLU.h"
#include "Sigmoid.h"
#include "MLP.h"
using namespace Hex;
using namespace std;

template <class T>
void printtensor(const Tensor<T>& tensor1) {
    tensor1.print();
}


int main() {


   
    int input_size = 4;        // Size of the input layer
    int output_size = 1;        // Size of the output layer
    int hiddenlayer = 1;        // Number of hidden layers
    int h_l_dimension = 2;     // Dimension of hidden layers

     
    MLP<float> mlp(input_size, output_size, hiddenlayer, h_l_dimension);

  
    Tensor<float> input_tensor({ input_size, 1 });
    initTensorOnGPU(input_tensor , 0.0f);
 
    Tensor<float> output_tensor = mlp.forward(input_tensor);
     mlp.backpropa(output_tensor , 0.001f);


   
    //DX_tensor.print();

    return 0;
}
