#include "CNN2D.h"
namespace Hex
{
    template<class T>
    Tensor<T>& CNN2D<T>::forward(Tensor<T>& input_tensor)
    {
        return input_tensor;
    }

    template<class T>
    Tensor<T>& CNN2D<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
    {
        return output_error;
    }

    template class CNN2D<float>;
    template class CNN2D<int>;
    template class CNN2D<double>;
}