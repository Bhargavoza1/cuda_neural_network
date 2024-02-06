#pragma once
#include "layer.h"

namespace Hex {
    template<class T>
    class linear : public layer<T>
    {
    private:
        bool _Isbias;
        bool _bias_as_zero;
        float _w_b_range;
        Tensor<T> weights;
        Tensor<T> bias; 
        Tensor<T> output;
        Tensor<T> input;
        Tensor<T> input_error;

    public:
        // Constructor
        linear(int input_size, int output_size, bool bias_as_zero = true , float w_b_range = 0.5f );
        ~linear();

        // Override forward method
        Tensor<T>& forward(Tensor<T>& tensor) override;

        // Override backpropagation method
        Tensor<T>& backpropagation(Tensor<T>& tensor, float learning_rate = 0.001f) override;

        Tensor<T>& printW();
        Tensor<T>& printB();
        //Tensor<T>& printO();


    private:
        void init_weight_n_bias();
   
    };
}

// Include the implementation in a separate .cpp file if necessary
 