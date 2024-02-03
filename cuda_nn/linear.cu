#include "linear.h"

namespace Hex{

	template<class T>
	linear<T>::linear(int input_size, int output_size, bool Isbias, float w_b_range)
		: _Isbias(Isbias) , _w_b_range(w_b_range),
		weights(std::vector<int>{output_size, input_size}),
		bias(Isbias ? Tensor<T>(std::vector<int>{1,output_size}) : Tensor<T>()),
		gradients_w(std::vector<int>{output_size, input_size}),
		gradients_b(Isbias ? Tensor<T>(std::vector<int>{output_size, 1}) : Tensor<T>())
	{
		 
	}


	template<class T>
	Tensor<T>& linear<T>::forward(Tensor<T>& tensor)
	{
		// TODO: insert return statement here
		return tensor;
	}

	template<class T>
	Tensor<T>& linear<T>::backpropagation(Tensor<T>& tensor, float learning_rate)
	{
		// TODO: insert return statement here
		return tensor;
	}
 
    // Explicit instantiation of the template class for supported types
    template class linear<float>;
    template class linear<int>;
    template class linear<double>;
}