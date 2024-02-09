#pragma once
#include "../utils/tensor.h"
namespace Hex{

	template <class T>
	class layer
	{
	private:

	public:
		virtual	~layer() = 0;

		virtual Tensor<T>& forward(Tensor<T>& input_tensor) = 0;
		virtual Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.0001f) = 0;

	};
}

template <class T>
inline Hex::layer<T>::~layer() {}
