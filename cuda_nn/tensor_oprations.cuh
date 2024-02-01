#pragma once

#ifndef TENSOROP_H
#define TENSOROP_H

#include <iostream>
#include "Tensor.h"

namespace Hex {

	template<class T, class U>
	std::unique_ptr<Tensor<typename std::common_type<T, U>::type>> addTensor(const Tensor<T>& tensor1, const Tensor<U>& tensor2);

	template<class T, class U>
	__global__ void addKernel(const T* a, const U* b, typename std::common_type<T, U>::type* c, int size);

	template <typename T>
	void initTensorOnGPU(Tensor<T>& tensor, float multiplier);

	template <typename T>
	__global__ void initializeTensor(T* data, int size, float multiplier);

}

#include "tensor_oprations.cu"
#endif  // TENSOROP_H