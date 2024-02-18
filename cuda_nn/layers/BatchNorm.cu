
#include "BatchNorm.h"
#include "../utils/tensor_oprations.h"
#include <cassert>
namespace Hex {

    template <class T>
    BatchNorm<T>::BatchNorm(int Batch_or_channels, TensorShape tensorshape , float momentum, float eps)
        : momentum(momentum), eps(eps) , _Tshape(tensorshape) ,
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
    Tensor<T>& BatchNorm<T>::forward(Tensor<T>& input_tensor, bool Istraining  ) {
        int tensor_dimensions = input_tensor.getShape().size();
        //std::cout << " size from batch norm forward : " << tensor_dimensions << std::endl;
        if (tensor_dimensions == 4 && _Tshape == TensorShape::_4D) {
            //gamma.print();
            return forward_4d(input_tensor, Istraining);
        }
        else if (tensor_dimensions == 2 && _Tshape == TensorShape::_2D) {
            return forward_2d(input_tensor, Istraining);
        }
        assert(false && "Invalid tensor dimensions or shape");
    }

    template <class T>
    Tensor<T>& BatchNorm<T>::backpropagation(Tensor<T>& output_error, float learning_rate) {
        return output_error;
    }

    template<class T>
    Tensor<T>& BatchNorm<T>::forward_2d(Tensor<T>& input_tensor, bool Istraining)
    {
        return input_tensor;
    }

    template<class T>
    Tensor<T>& BatchNorm<T>::forward_4d(Tensor<T>& input_tensor, bool Istraining)
    {
        return input_tensor;
    }

    template class BatchNorm<float>;
    template class BatchNorm<int>;
    template class BatchNorm<double>;
}