#pragma once
 
#include "layer.h"
#include <iostream>
#include <memory>
namespace Hex {


    enum class TensorShape {
        _2D,
        _4D
    };

    template <class T>
    class BatchNorm : public layer<T>
    {
    private:
        
        float momentum;
        float eps;

        
        Tensor<T> gamma;  
        Tensor<T> beta;  
        Tensor<T> running_mean;
        Tensor<T> running_var;

     
        Tensor<T> x_normalized;
        Tensor<T> input_mean;
        Tensor<T> input_var;
        Tensor<T> input;


        std::unique_ptr<Tensor<T>> input_error;

        std::unique_ptr<Tensor<T>> output;

    

    public:
        BatchNorm(int Batch_or_channels,TensorShape tensorshape = TensorShape::_4D ,float momentum = 0.9, float eps = 1e-5);
        
        ~BatchNorm();

        Tensor<T>& forward(Tensor<T>& input_tensor) override;
        Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.0001f) override;
    };



} 
