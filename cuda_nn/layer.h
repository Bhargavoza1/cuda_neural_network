#pragma once

namespace Hex{
	class layer
	{
	public:
		virtual	~layer() = 0;

		virtual void forward() = 0;
		virtual void backpropagation() = 0;

	};
}


