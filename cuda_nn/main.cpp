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
    //int batchsize = 64;
    //int input_channels = 3;
    //int output_class = 2;
    //std::unique_ptr<Hex::Image_CF<float>>  Image_CF(new  Hex::Image_CF<float>(batchsize, input_channels, output_class));

    //std::vector<int> x_shape = { batchsize,input_channels,512 ,512 }; // Shape: (4, 1, 2)


    //std::unique_ptr<Tensor<float>> x_tensor(new Tensor<float>(x_shape));
    //initTensorOnGPU(*x_tensor, 0.0f);
    // x_tensor->print();

    //Tensor<float>* a = &Image_CF->forward(*x_tensor);   a->print();
    // a = &Image_CF->forward(*x_tensor);   a->print();
    //  a = &Image_CF->forward(*x_tensor);   a->print();
 

   // Image_CF->backpropa(*a);
 

    int batchsize = 64;
    int input_channels = 3 ;
    int output_class = 2;
    std::unique_ptr<Hex::Image_CF<float>>  Image_CF(new  Hex::Image_CF<float>(batchsize, input_channels, output_class));

    std::vector<int> x_shape = { batchsize,input_channels,256 ,256 }; // Shape: (4, 1, 2)

  
    std::unique_ptr<Tensor<float>> x_tensor(new Tensor<float>(x_shape));
    initTensorOnGPU(*x_tensor, 0.0f);
     //  x_tensor->print();
    Tensor<float> a;

    for (int i = 0; i < 10; i++) {
        a = Image_CF->forward(*x_tensor);
        Image_CF->backpropa(a);
        std::cout << " epoch " << i << " done" << std::endl;
    }

    

    

    // CNN2D<float> cn1(batchsize, { input_channels  ,output_class }, 3 , 0);
    // auto a = cn1.forward(*x_tensor);
    // auto b = cn1.backpropagation(a);
    // a.print();
    //b.print();
    return 0;

}
