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
        // input channel = 3 , output channel = 16 , kernel size = 3 is 3x3
        conv1(batch_size, {3,16}, 3) , 
        // input channel = 16  
        bn1(16, TensorShape::_4D),
        // input channel = 16 , output channel = 32 , kernel size = 3 is 3x3
        conv2(batch_size, {16,32}, 3),
        // kernel size = 2 is 2x2
        pool(2),
        relu1(),
        fl(),
        linear1( 32 * 64 * 64 , 128, batch_size),
        linear2( 128, 64, batch_size),
        linear3( 64, 2, batch_size),
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

       x = conv1.forward(input_tensor, Istraining);
       x = relu1.forward(x, Istraining);
       x = bn1.forward(x, Istraining);
       x = pool.forward(x, Istraining);
       x = conv2.forward(x, Istraining);
       x = relu1.forward(x, Istraining);
     //  x = bn1.forward(x, Istraining);
       x = pool.forward(x, Istraining);

       x = fl.forward(x, Istraining);

 
       x = linear1.forward(x, Istraining);
       x = relu1.forward(x, Istraining);
       x = linear2.forward(x, Istraining);
       x = relu1.forward(x, Istraining);
       x = linear3.forward(x, Istraining);
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
        x = sigmoid1.backpropagation(x, learning_rate);
        x = linear3.backpropagation(x, learning_rate);
        x = relu1.backpropagation(x, learning_rate);
        x = linear2.backpropagation(x, learning_rate);
        x = relu1.backpropagation(x, learning_rate);
        x = linear1.backpropagation(x, learning_rate);

        x = fl.backpropagation(x, learning_rate);

        x = pool.backpropagation(x, learning_rate);
        x = relu1.backpropagation(x, learning_rate);
        x = conv2.backpropagation(x, learning_rate);
        x = pool.backpropagation(x, learning_rate);
        x = bn1.backpropagation(x, learning_rate);
        x = relu1.backpropagation(x, learning_rate);
        conv1.backpropagation(x, learning_rate);
    }
}