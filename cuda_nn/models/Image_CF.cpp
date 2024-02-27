#include "Image_CF.h"
#include "../layers/layer.h" 
#include "../layers/linear.h"
#include "../layers/ReLU.h"
#include "../layers/Sigmoid.h"
#include "../layers/BatchNorm.h"
#include "../utils/Tensor.h"

namespace Hex {

    template<class T>
    inline Image_CF<T>::Image_CF(int batch_size , int input_channels, int output_class):
        // input channel = input_channels (1 or 3) , output channel = 16 , kernel size = 3 is 3x3
        conv1(batch_size, { input_channels,16}, 3) ,
        relu1(),
        // input channel = 16  
        bn1(16, TensorShape::_4D),
        // kernel size = 2 is 2x2 , stride is = 2
        pool1(2, 3),

        // input channel = 16 , output channel = 32 , kernel size = 3 is 3x3
        conv2(batch_size, {16,32}, 3),
        relu2(),
        bn2(32, TensorShape::_4D),
        pool2(2, 3),

        fl(),

        linear1( 32 * 57 * 57, 128, batch_size),
        relu3(),
        bn3(128, TensorShape::_2D),

        linear2(128, output_class, batch_size),
        sigmoid1() 
    {
    }

    template<class T>
    Image_CF<T>::~Image_CF()
    {
    }
    template<class T>
    Tensor<T>& Image_CF<T>::forward(Tensor<T>& input_tensor, bool Istraining)
    {
      
        //input_tensor.print();
        std::cout<<"init done" ;
       x = conv1.forward(input_tensor, Istraining);
       //x.print();
       //x.print();
       x = relu1.forward(x, Istraining);
      
       // x.print();
       // x = bn1.forward(x, Istraining);
       // x.print();
       x = pool1.forward(x, Istraining);
       // x.print();
       x = conv2.forward(x, Istraining);
       //x.print();
       x = relu2.forward(x, Istraining);
       //x.print();
         x = bn2.forward(x, Istraining);
       x = pool2.forward(x, Istraining);
        //x.print();
       //   x.print();
        x = fl.forward(x, Istraining);
      
 
       x = linear1.forward(x, Istraining);
      // //x.print();
       x = relu3.forward(x, Istraining);
         x = bn3.forward(x, Istraining);
       //x.print();
        x = linear2.forward(x, Istraining);
        //x.print();
      // x = relu1.forward(x, Istraining);
       //x = linear3.forward(x, Istraining);
       x = sigmoid1.forward(x, Istraining);

        return x;
    }
    template<class T>
    Tensor<T>& Image_CF<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
    {
        return output_error;
    }
    template<class T>
    void Image_CF<T>::backpropa(Tensor<T>& output_error, float learning_rate)
    {
        x = output_error;
       // 
        x = sigmoid1.backpropagation(x, learning_rate);
       // x = linear3.backpropagation(x, learning_rate);
       // x = relu1.backpropagation(x, learning_rate);
       
        x = linear2.backpropagation(x, learning_rate);
        
        x = bn3.backpropagation(x, learning_rate);
      
        x = relu3.backpropagation(x, learning_rate);
        //x.print();
         x = linear1.backpropagation(x, learning_rate);
       // 
        x = fl.backpropagation(x, learning_rate);
       
        x = pool2.backpropagation(x, learning_rate);
        //x.print();
          x = bn2.backpropagation(x, learning_rate);
        //  x.print();
        x = relu2.backpropagation(x, learning_rate);
        
        x = conv2.backpropagation(x, learning_rate);
        //  x.print();
       // x.print();
        x = pool1.backpropagation(x, learning_rate);
       
        // x = bn1.backpropagation(x, learning_rate);
         
        x = relu1.backpropagation(x, learning_rate);
       
        x = conv1.backpropagation(x, learning_rate);
        //x.print();
        
    }

    // Explicit instantiation of the template class for supported types
    template class Image_CF<float>;
    template class Image_CF<int>;
    template class Image_CF<double>;
}