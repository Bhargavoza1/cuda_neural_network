#pragma once
#include "layer.h"

namespace Hex {
	template<class T>
	class linear : public layer<T>
	{

	public:
		Tensor<T>& forward(Tensor<T>& tensor) override;
		Tensor<T>& backpropagation(Tensor<T>& tensor, float learning_rate = 0.0001f) override;
	};
	
}
 

