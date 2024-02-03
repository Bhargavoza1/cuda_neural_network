#pragma once
#include "layer.h"

namespace Hex {
    template<class T>
    class linear : public layer<T>
    {
    private:
        bool _Isbias;
        float _w_b_range;
        Tensor<T> weights;
        Tensor<T> bias;
        Tensor<T> gradients_w;
        Tensor<T> gradients_b;

    public:
        // Constructor
        linear(int input_size, int output_size, bool Isbias = true, float w_b_range = 0.5f);


        // Override forward method
        Tensor<T>& forward(Tensor<T>& tensor) override;

        // Override backpropagation method
        Tensor<T>& backpropagation(Tensor<T>& tensor, float learning_rate = 0.0001f) override;
    };
}

// Include the implementation in a separate .cpp file if necessary
 