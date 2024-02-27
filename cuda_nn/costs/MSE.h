#pragma once
#ifndef MSE_H
#define MSE_H
 
 
#include "../utils/tensor.h"

namespace Hex {

    template<typename T>
    std::shared_ptr<Tensor<T>> mse(Tensor<T>& y_true, Tensor<T>& y_pred);

    template<typename T>
    std::shared_ptr<Tensor<T>> mse_derivative(Tensor<T>& y_true, Tensor<T>& y_pred);
}

#include "MSE.cu"
#endif  // MSE_H