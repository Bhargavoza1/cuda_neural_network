#include <iostream>
#include "example/XOR.h"
#include "models/Image_CF.h"
#include "utils/Tensor.h"
#include "utils/tensor_oprations.h"
#include "layers/BatchNorm.h"
#include "layers/linear.h"
using namespace std;
using namespace Hex;
#include "layers/CNN2D.h"

int main() {

    //   Hex::xor_example();
    int batchsize = 64;
    int input_channels = 3;
    int output_class = 2;
    std::unique_ptr<Hex::Image_CF<float>>  Image_CF(new  Hex::Image_CF<float>(batchsize, input_channels, output_class));

    std::vector<int> x_shape = { batchsize,input_channels,512 ,512 }; // Shape: (4, 1, 2)


    std::unique_ptr<Tensor<float>> x_tensor(new Tensor<float>(x_shape));
    initTensorOnGPU(*x_tensor, 0.0f);
    // x_tensor->print();

    Tensor<float>* a = &Image_CF->forward(*x_tensor);
    a->print();

    Image_CF->backpropa(*a);
 

 
    return 0;

}
