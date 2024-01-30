#pragma once
#include "layer.h"

namespace Hex {
	class linear : public layer
	{
		void forward() override;
		void backpropagation() override;
	};
}
 

