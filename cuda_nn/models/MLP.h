#pragma once
#include "../layers/layer.h"
#include "MLP.h"
#include "../layers/linear.h"
#include "../layers/ReLU.h"
#include "../layers/Sigmoid.h"
#include "../layers/BatchNorm.h"
#include "../utils/Tensor.h"
namespace Hex {
	template <class T>
	class MLP : public layer<T>
	{
	private:

        int _hiddenlayer;
        Tensor<T> X;
       
        linear<T>  linear1;
        BatchNorm<T> bn1;
        ReLU<T>  relu1;
        linear<T>  linear2;
        ReLU<T>  relu2;
        linear<T>  linear3;
        Sigmoid<T>  sigmoid1;

	public:
        // Constructor
        MLP(int input_size, int output_size,  int batch_size = 1 , int hiddenlayer = 1, int h_l_dimension = 10);
        ~MLP();

        // Override forward method
        Tensor<T>& forward(Tensor<T>& input_tensor, bool Istraining = true) override;

        // Override backpropagation method
        Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.001f) override;

        void backpropa(Tensor<T>& output_error, float learning_rate = 0.001f)  ;
	};
}


