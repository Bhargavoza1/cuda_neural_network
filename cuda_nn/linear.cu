#include "linear.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
namespace Hex{

	template<class T>
	__global__ void initWeightKernel(T* weights, int output_size, int input_size, float w_b_range) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		if (i < output_size && j < input_size) {
			// Random initialization of weights within the specified range
			curandState state;
			curand_init(clock64(), i * input_size + j, 0, &state);

			float float_weight = (2 * curand_uniform(&state) - 1) * w_b_range;
			weights[i * input_size + j] = static_cast<T>(float_weight);
		}
	}

	template<class T>
	linear<T>::linear(int input_size, int output_size, bool Isbias, float w_b_range)
		: _Isbias(Isbias) , _w_b_range(w_b_range),
		weights(std::vector<int>{output_size, input_size}),
		bias(Isbias ? Tensor<T>(std::vector<int>{1,output_size}) : Tensor<T>()),
		gradients_w(std::vector<int>{output_size, input_size}),
		gradients_b(Isbias ? Tensor<T>(std::vector<int>{output_size, 1}) : Tensor<T>())
	{
		initweight();
		initbias();
	}


	template<class T>
	Tensor<T>& linear<T>::forward(Tensor<T>& tensor)
	{
		// TODO: insert return statement here
		return weights;
	}

	template<class T>
	Tensor<T>& linear<T>::backpropagation(Tensor<T>& tensor, float learning_rate)
	{
		// TODO: insert return statement here
		return tensor;
	}

	template<class T>
	void linear<T>::initweight() {
		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks((weights.getShape()[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(weights.getShape()[0] + threadsPerBlock.y - 1) / threadsPerBlock.y);

		// Launch the kernel to initialize weights
		initWeightKernel << <numBlocks, threadsPerBlock >> > (weights.getData(), weights.getShape()[0], weights.getShape()[1], _w_b_range);
		cudaDeviceSynchronize();  // Wait for the kernel to finish
	}

	template<class T>
	void linear<T>::initbias()
	{
	}
 
    // Explicit instantiation of the template class for supported types
    template class linear<float>;
    template class linear<int>;
    template class linear<double>;
}