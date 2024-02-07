#pragma once

#ifndef TENSOROP_H
#define TENSOROP_H

#include <iostream>
#include "Tensor.h"

namespace Hex {

	template<class T, class U>
	std::unique_ptr<Tensor<typename std::common_type<T, U>::type>> addTensor(const Tensor<T>& tensor1, const Tensor<U>& tensor2);

	template <typename T>
	void initTensorOnGPU(Tensor<T>& tensor, float multiplier);

}

#include "tensor_oprations.cu"
#endif  // TENSOROP_H