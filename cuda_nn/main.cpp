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

        Hex::xor_example();
     int batchsize = 32;
    int input_channels = 3;
    int output_class = 2;
    std::unique_ptr<Hex::Image_CF<float>>  Image_CF(new  Hex::Image_CF<float>( batchsize, input_channels , output_class));
   
    std::vector<int> x_shape = { batchsize,input_channels,512 ,512 }; // Shape: (4, 1, 2)
  

    std::unique_ptr<Tensor<float>> x_tensor(new Tensor<float>(x_shape));
    initTensorOnGPU(*x_tensor , 0.0f);
   // x_tensor->print();
 
    Tensor<float>* a = &Image_CF->forward(*x_tensor);
     a->print();
          
      Image_CF->backpropa(*a);          
            a = &Image_CF->forward(*x_tensor);
     a->print();
          
      Image_CF->backpropa(*a);          
          
                   a = &Image_CF->forward(*x_tensor);
     a->print();
          
      Image_CF->backpropa(*a);          
          
                   a = &Image_CF->forward(*x_tensor);
     a->print();
          
      Image_CF->backpropa(*a);          
          
       
      //  a->print();
       // 
       // 
  //  CNN2D<float> cn1(batchsize, { input_channels  ,output_class }, 3);
  //  BatchNorm<float> bn1(output_class, TensorShape::_4D);
  ////  Tensor<float> a = cn1.forward(*x_tensor);
  //// a.print();
  //  Tensor<float> b = bn1.forward(*x_tensor);
  //  Tensor<float> c = bn1.backpropagation(b);

  //  c.print();
     // std::vector<int> x_shape = { batchsize,input_channels  };
     // std::shared_ptr<Tensor<float>> x_tensor(std::make_shared<Tensor<float>>(x_shape));
     //Tensor<float> y_tensor;
     // initTensorOnGPU(*x_tensor, 0.0f);
     //linear<float> l1(input_channels , output_class, batchsize);
     //auto a =  l1.forward(*x_tensor);
     //y_tensor = *x_tensor;
     //y_tensor.print();
    return 0;

}
