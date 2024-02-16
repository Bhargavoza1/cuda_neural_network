#pragma once
 
#include "layer.h"
#include <iostream>
#include <memory>
namespace Hex {

    template <class T>
    class BatchNorm : public layer<T>
    {
    private:
        // Define your member variables for BatchNorm
        // Batch normalization parameters
        Tensor<T> gamma; // Scale parameter
        Tensor<T> beta; // Shift parameter
        Tensor<T> running_mean;
        Tensor<T> running_var;

        // Cache for backpropagation
        Tensor<T> x_normalized;
        Tensor<T> input_mean;
        Tensor<T> input_var;
        Tensor<T> input;

        // Momentum and epsilon for running statistics
        float momentum;
        float eps;

    public:
        BatchNorm(int channels, float momentum = 0.9, float eps = 1e-5); 
        ~BatchNorm();

        Tensor<T>& forward(Tensor<T>& input_tensor) override;
        Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.0001f) override;
    };



} 
