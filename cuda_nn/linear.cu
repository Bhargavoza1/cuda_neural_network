#include "linear.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
namespace Hex{

	template<class T>
	__global__ void initWeightKernel(T* weights, T* bias, int output_size, int input_size, bool bias_as_zero, float w_b_range, bool Isbias) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		if (i < output_size && j < input_size) {
			// Random initialization of weights within the specified range
			curandState state;
			curand_init(clock64(), i * input_size + j, 0, &state);

			float float_weight = (2 * curand_uniform(&state) - 1) * w_b_range;
			weights[i * input_size + j] = static_cast<T>(float_weight);
		}

		// Initialize bias if Isbias is true
		if (Isbias && i < output_size && j == 0) {
			if (bias_as_zero) {
				 
				bias[i] = static_cast<T>(0);
			}
			else {
				curandState state_bias;
				curand_init(clock64(), i, 0, &state_bias);

				float float_bias = (2 * curand_uniform(&state_bias) - 1) * w_b_range;
				bias[i] = static_cast<T>(float_bias);
			}
			
		}
	}

	template<class T>
	linear<T>::linear(int input_size, int output_size,bool bias_as_zero, float w_b_range, bool Isbias)
		: _bias_as_zero(bias_as_zero), _w_b_range(w_b_range), _Isbias(Isbias),
		weights(std::vector<int>{ input_size,output_size }),
		bias(Isbias ? Tensor<T>(std::vector<int>{1,output_size}) : Tensor<T>()),
		gradients_w(std::vector<int>{input_size, output_size}),
		gradients_b(Isbias ? Tensor<T>(std::vector<int>{1, output_size}) : Tensor<T>())
	{
		init_weight_n_bias();
	}


	template<class T>
	Tensor<T>& linear<T>::forward(Tensor<T>& tensor)
	{
		// TODO: insert return statement here
		return bias;
	}

	template<class T>
	Tensor<T>& linear<T>::backpropagation(Tensor<T>& tensor, float learning_rate)
	{
		// TODO: insert return statement here
		return tensor;
	}

	template<class T>
	void linear<T>::init_weight_n_bias() {
		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks((weights.getShape()[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(weights.getShape()[0] + threadsPerBlock.y - 1) / threadsPerBlock.y);

		// Launch the kernel to initialize weights and bias
		initWeightKernel << <numBlocks, threadsPerBlock >> > (weights.getData(), bias.getData(), weights.getShape()[0],
															 weights.getShape()[1], _bias_as_zero, _w_b_range, _Isbias);
		cudaDeviceSynchronize();  // Wait for the kernel to finish
	}

 
 
    // Explicit instantiation of the template class for supported types
    template class linear<float>;
    template class linear<int>;
    template class linear<double>;
}