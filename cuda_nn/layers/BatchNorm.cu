
#include "BatchNorm.h"
namespace Hex {

    template <class T>
    BatchNorm<T>::BatchNorm(int channels, float momentum, float eps) 
        : momentum(momentum), eps(eps) {
  
        gamma = Tensor<T>({ 1, channels, 1, 1 }); // Initialize to ones
        beta = Tensor<T>({ 1, channels, 1, 1 }); // Initialize to zeros
        running_mean = Tensor<T>({ 1, channels, 1, 1 }); // Initialize to zeros
        running_var = Tensor<T>({ 1, channels, 1, 1 }); // Initialize to ones

        gamma.print();
    }

    template <class T>
    BatchNorm<T>::~BatchNorm() {
        // Destructor implementation
    }

    template <class T>
    Tensor<T>& BatchNorm<T>::forward(Tensor<T>& input_tensor) {
        return input_tensor;
    }

    template <class T>
    Tensor<T>& BatchNorm<T>::backpropagation(Tensor<T>& output_error, float learning_rate) {
        return output_error;
    }

    template class BatchNorm<float>;
    template class BatchNorm<int>;
    template class BatchNorm<double>;
}