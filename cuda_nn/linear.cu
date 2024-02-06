#include "linear.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include<iostream>
#include "Errorhelper.cpp"
namespace Hex{

	template<class T>
	__global__ void initWeightKernel(T* weights, T* bias, int output_size, int input_size, bool bias_as_zero, float w_b_range, bool Isbias) {
		//int i = blockIdx.x * blockDim.x + threadIdx.x;
		//int j = blockIdx.y * blockDim.y + threadIdx.y;

		//if (i < output_size && j < input_size) {
		//	 Random initialization of weights within the specified range
		//	curandState state;
		//	curand_init(clock64(), i * input_size + j, 0, &state);

		//	float float_weight = (2 * curand_uniform(&state) - 1) * w_b_range;
		//	weights[i * input_size + j] = static_cast<T>(float_weight);
		//}

		// Initialize bias if Isbias is true
		//if (Isbias && i < output_size && j == 0) {
		//	if (bias_as_zero) {
		//		 
		//		bias[i] = static_cast<T>(0.0);
		//	}
		//	else {
		//		curandState state_bias;
		//		curand_init(clock64(), i, 0, &state_bias);

		//		float float_bias = (2 * curand_uniform(&state_bias) - 1) * w_b_range;
		//		bias[i] = static_cast<T>(float_bias);
		//	}
		//	
		//}

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		if (i < output_size && j < input_size) {
			 
			weights[i * input_size + j] = static_cast<T>(i * input_size + j + 1);   
		}

		// Initialize bias if Isbias is true
		if (Isbias && i < output_size && j == 0) {
			if (bias_as_zero) {
				bias[i] = static_cast<T>(0.0);
			}
			else {
				 
				bias[i] = static_cast<T>(i + 1);   
			}
		}
	}

	template<class T>
	__global__ void linearLayerForward(const T* W, const T* X, T* Y, const T* b,
		int W_x_dim, int W_y_dim,
		int X_x_dim, int X_y_dim) {

		int col = blockIdx.y * blockDim.y + threadIdx.y;
		int row = blockIdx.x * blockDim.x + threadIdx.x;
	

		int Y_x_dim = W_x_dim;
		int Y_y_dim = X_y_dim;

		T Y_value = 0;

		if (row < Y_x_dim && col < Y_y_dim) {
			// Perform the matrix multiplication: Y = W * A  
			for (int i = 0; i < W_y_dim; ++i) {
				Y_value += W[row * W_y_dim + i] * X[i]; 
				//printf("W[row * W_y_dim + i] %d\n", W[row * W_y_dim + i]);
				 //	printf("W[row * W_x_dim + i] %d\n", W[i * W_x_dim + row]);
				//Y_value += W[row * W_y_dim + i] * X[i * X_y_dim + col];
			}
	
			// Add bias Y_value + b
			Y_value += b[row];

			// Store the result in the output tensor
			Y[row * Y_y_dim + col] = Y_value;
		}
 

	}


	template<class T>
	linear<T>::linear(int input_size, int output_size,bool bias_as_zero, float w_b_range, bool Isbias)
		: _bias_as_zero(bias_as_zero), _w_b_range(w_b_range), _Isbias(Isbias),
		weights(std::vector<int>{output_size , input_size  }),
		bias(Isbias ? Tensor<T>(std::vector<int>{output_size,1}) : Tensor<T>()), 
		output(std::vector<int>{output_size, 1  }),
		input(std::vector<int>{input_size, 1  }),
		input_error(std::vector<int>{input_size, 1  })
	{
		init_weight_n_bias();
	}


	template<class T>
	Tensor<T>& linear<T>::forward(Tensor<T>& tensor)
	{
		input = tensor;
		if (weights.getShape()[1] != input.getShape()[0]) {
			std::cerr << "Error: Tensor shapes must be the same for addition. Shape of tensor1: "
				<< weights.getShape()[1] << ", Shape of tensor2: " << input.getShape()[0] << std::endl;
			throw std::runtime_error("Tensor shape mismatch");
		}

		// Ensure dimensions match
		//assert(tensor.getShape()[1] == weights.getShape()[0]);

		// Allocate memory for the output tensor
		//std::cout << weights.getShape()[0] << "X" << tensor.getShape()[1] << std::endl;
		 

		dim3 threadsPerBlock(256);
		dim3 numBlocks((output.getShape()[0] + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(output.getShape()[1] + threadsPerBlock.y - 1) / threadsPerBlock.y);
		// Launch the forward kernel
		 
		linearLayerForward << <numBlocks, threadsPerBlock >> > (weights.getData(), input.getData(), output.getData(), bias.getData(),
			weights.getShape()[0], weights.getShape()[1] ,
			input.getShape()[0], input.getShape()[1]);
		cudaDeviceSynchronize();
 ;
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			printf("CUDA error from add tensor: %s\n", cudaGetErrorString(cudaError));
			exit(EXIT_FAILURE);  // or handle the error appropriately
		}

		return output;
	}

	template<class T>
	__global__ void backpropagationAndUpdateKernel(T* weights, T* bias,
		const T* input_gradients,const T* input_data, T* input_error,
		float learning_rate, int w_x_dim, int w_y_dim,
		int input_x_dim, int input_y_dim)
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		int col = blockIdx.y * blockDim.y + threadIdx.y;
		
		if (row < w_y_dim && col < input_y_dim) {
			T sum = 0;
			for (int i = 0; i < w_x_dim; ++i) {
				sum  += weights[i * w_y_dim + row] * input_gradients[i]; 
			} 
			input_error[row * input_y_dim + col] = sum; 
		}

		if (row < w_x_dim && col < input_y_dim) {
			T gw = 0;
			
			bias[row] -= learning_rate * input_gradients[row];
			for (int i = 0; i < w_y_dim; ++i) {
				 gw  = input_gradients[row] * input_data[i]; 
				 weights[row * w_y_dim + i] -= learning_rate * gw;
			}
			
		}
	}

	template<class T>
	Tensor<T>& linear<T>::backpropagation(Tensor<T>& input_gradients, float learning_rate)
	{
		
		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks((weights.getShape()[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(weights.getShape()[0] + threadsPerBlock.y - 1) / threadsPerBlock.y);

		//std::cout << weights.getShape()[1] << "aaaX" << input_gradients.getShape()[1] << std::endl;
		backpropagationAndUpdateKernel << <numBlocks, threadsPerBlock >> > (
			weights.getData(), bias.getData(),
			input_gradients.getData(), input.getData(), input_error.getData(),
			learning_rate, weights.getShape()[0], weights.getShape()[1],
			input_gradients.getShape()[0], input_gradients.getShape()[1]);
		cudaDeviceSynchronize();

		return input_error;


	}



	template<class T>
	void linear<T>::init_weight_n_bias() {
		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks((weights.getShape()[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(weights.getShape()[0] + threadsPerBlock.y - 1) / threadsPerBlock.y);

		// Launch the kernel to initialize weights and bias
		initWeightKernel << <numBlocks, threadsPerBlock >> > (weights.getData(), bias.getData(), weights.getShape()[0],
															 weights.getShape()[1], _bias_as_zero, _w_b_range, _Isbias);
		cudaDeviceSynchronize();   
	}

 

	template<class T>
	Tensor<T>& linear<T>::printW()
	{
		return weights;
	}

	template<class T>
	Tensor<T>& linear<T>::printB()
	{
		return bias;
	}
 
    // Explicit instantiation of the template class for supported types
    template class linear<float>;
    template class linear<int>;
    template class linear<double>;
}