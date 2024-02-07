#pragma once
#include "layer.h"
#include <iostream>
namespace Hex{
	template <class T>
	class ReLU : public layer<T>
	{
    private:
 
        std::unique_ptr<Tensor<T>> output;
        std::unique_ptr<Tensor<T>> input;
        std::unique_ptr<Tensor<T>> input_error;

    public:
        // Constructor
        ReLU();
        ~ReLU();

        // Override forward method
        Tensor<T>& forward(Tensor<T>& tensor) override;

        // Override backpropagation method
        Tensor<T>& backpropagation(Tensor<T>& tensor, float learning_rate = 0.001f) override;
 

    private:
        

    };

}



