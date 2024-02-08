#pragma once
#include "layer.h"
#include <iostream>
#include <cmath>

namespace Hex {
    template <class T>
    class Sigmoid : public layer<T> {
    private:

        Tensor<T> input;
        std::unique_ptr<Tensor<T>> output;
        std::unique_ptr<Tensor<T>> input_error;

    public:
        // Constructor
        Sigmoid();

        // Destructor
        ~Sigmoid();

        // Override forward method
        Tensor<T>& forward(Tensor<T>& input_tensor) override;

        // Override backpropagation method
        Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.001f) override;
    };
}


