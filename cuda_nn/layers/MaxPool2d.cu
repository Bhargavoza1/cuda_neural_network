
#include"MaxPool2d.h"

namespace Hex {
	template<class T>
	MaxPool2d<T>::MaxPool2d(int kernel_size, int padding  , int stride  )
	{
	}
	template<class T>
	MaxPool2d<T>::~MaxPool2d()
	{
	}
	template<class T>
	Tensor<T>& MaxPool2d<T>::forward(Tensor<T>& input_tensor)
	{
		return input_tensor;
	}
	template<class T>
	Tensor<T>& MaxPool2d<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
	{
		return output_error;
	}
}