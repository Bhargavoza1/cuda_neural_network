#pragma once
#include "layer.h"
namespace Hex {
	template <class T>
	class MaxPool2d : public layer<T>
	{
	private:


	public:
		MaxPool2d(int kernel_size, int padding = 1, int stride = 1);
		~MaxPool2d();


		// Override forward method
		Tensor<T>& forward(Tensor<T>& input_tensor) override;

		// Override backpropagation method
		Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.001f) override;
	private:

	};

}