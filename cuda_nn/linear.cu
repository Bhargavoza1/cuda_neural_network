#include "linear.h"

namespace Hex{

	template<class T>
	Tensor<T>& linear<T>::forward(Tensor<T>& tensor)
	{
		// TODO: insert return statement here
	}
	template<class T>
	Tensor<T>& linear<T>::backpropagation(Tensor<T>& tensor, float learning_rate = 0.0001f)
	{
		// TODO: insert return statement here
	}
    // Explicit instantiation of the template class for supported types
    template class Tensor<float>;
    template class Tensor<int>;
    template class Tensor<double>;
}