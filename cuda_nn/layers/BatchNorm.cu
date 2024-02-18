
#include "BatchNorm.h"
#include "../utils/tensor_oprations.h"
namespace Hex {

    template <class T>
    BatchNorm<T>::BatchNorm(int Batch_or_channels, TensorShape tensorshape , float momentum, float eps)
        : momentum(momentum), eps(eps) ,
        gamma(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, Batch_or_channels, 1, 1 }) : Tensor<T>({ 1, Batch_or_channels })),
        beta(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, Batch_or_channels, 1, 1 }) : Tensor<T>({ 1, Batch_or_channels })),
        running_mean(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, Batch_or_channels, 1, 1 }) : Tensor<T>({ 1, Batch_or_channels })),
        running_var(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, Batch_or_channels, 1, 1 }) : Tensor<T>({ 1, Batch_or_channels }))
    {

        initTensorToOneOnGPU(gamma);
        initTensorToOneOnGPU(running_var); 
   
        
    }

    template <class T>
    BatchNorm<T>::~BatchNorm() {
        // Destructor implementation
    }

    template <class T>
    Tensor<T>& BatchNorm<T>::forward(Tensor<T>& input_tensor) {
        gamma.print();
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