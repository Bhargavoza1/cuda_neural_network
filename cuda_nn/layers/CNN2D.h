#pragma once
#include "layer.h"

namespace Hex
{
    template <class T>
    class CNN2D : public layer<T>
    {
    public:
       Tensor<T>& forward(Tensor<T>& input_tensor) override;
       Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.0001f)override;
    };

 

}
