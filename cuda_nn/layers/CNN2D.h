#pragma once
#include "layer.h"
#include <iostream>
namespace Hex
{
    template <class T>
    class CNN2D : public layer<T>
    {
    private:
        int _batch_size;
        int _in_channels;
        int _out_channels;
        int _kernel_size;
        int _padding;
        int _stride;
        float _w_b_range;
        Tensor<T> weights;
        Tensor<T> bias;
       
        Tensor<T> input;
        Tensor<T> input_error;

        std::unique_ptr<Tensor<T>> output;
    public:
        CNN2D(const std::vector<int>& batch_height_width, const std::vector<int>& in_out_channels, int kernel_size, int padding = 0 , int stride = 1, float w_b_range = 0.5f);
        ~CNN2D();
       Tensor<T>& forward(Tensor<T>& input_tensor) override;
       Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.0001f)override;

    private:
        void init_weight_n_bias();
    };

 

}
