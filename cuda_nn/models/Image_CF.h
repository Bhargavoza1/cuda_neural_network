#pragma once
#include "../layers/layer.h" 
#include "../layers/linear.h"
#include "../layers/ReLU.h"
#include "../layers/Sigmoid.h"
#include "../layers/BatchNorm.h"
#include "../layers/CNN2D.h"
#include "../layers/flatten_layer.h"
#include "../layers/MaxPool2d.h"
#include "../utils/Tensor.h"

namespace Hex {
    template <class T>
    class Image_CF : public layer<T>
    {
    private:

        
        Tensor<T> X;

        CNN2D<T>  conv1;
        BatchNorm<T> bn1;
        CNN2D<T> conv2;
        MaxPool2d<T> pool; 
        ReLU<T>  relu1;
        flatten_layer<T> fl;
        linear<T>  linear1; 
        linear<T>  linear2;
        linear<T>  linear3;
        Sigmoid<T>  sigmoid1;

    public:
        // Constructor
        Image_CF(int batch_size = 1, int input_channels = 3 , int output_class = 2 );
        ~Image_CF();

        // Override forward method
        Tensor<T>& forward(Tensor<T>& input_tensor, bool Istraining = true) override;

        // Override backpropagation method
        Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.001f) override;

        void backpropa(Tensor<T>& output_error, float learning_rate = 0.001f);
    };

}


