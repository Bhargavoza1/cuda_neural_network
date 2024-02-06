#include "ReLU.h"

namespace Hex {
    template<class T>
    ReLU<T>::ReLU()
    {
    }
    template<class T>
    ReLU<T>::~ReLU()
    {
    }
    template<class T>
    inline Tensor<T>& ReLU<T>::forward(Tensor<T>& tensor)
    {
        output = Tensor<T>(tensor.getShape());
        input = Tensor<T>(tensor.getShape());
        input_error = Tensor<T>(tensor.getShape());
        return tensor;
        // TODO: insert return statement here
    }
    template<class T>
    Tensor<T>& ReLU<T>::backpropagation(Tensor<T>& tensor, float learning_rate)
    {
        return tensor;
        // TODO: insert return statement here
    }

    // Explicit instantiation of the template class for supported types
    template class ReLU<float>;
    template class ReLU<int>;
    template class ReLU<double>;
}