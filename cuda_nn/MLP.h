#pragma once
#include "layer.h"
#include "MLP.h"
#include "linear.h"
#include "ReLU.h"
#include "Sigmoid.h"
namespace Hex {
	template <class T>
	class MLP : public layer<T>
	{
	private:

        int _hiddenlayer;
        Tensor<T> X;
       
        linear<T>  linear1;
        ReLU<T>  relu1;
        linear<T>  linear2;
        ReLU<T>  relu2;
        linear<T>  linear3;
        Sigmoid<T>  sigmoid1;

	public:
        // Constructor
        MLP(int input_size, int output_size, int hiddenlayer = 1, int h_l_dimension = 10);
        ~MLP();

        // Override forward method
        Tensor<T>& forward(Tensor<T>& input_tensor) override;

        // Override backpropagation method
        Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.001f) override;

        void backpropa(Tensor<T>& output_error, float learning_rate = 0.001f)  ;
	};
}


