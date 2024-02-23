#include "Sigmoid.h"
#include <cuda_runtime.h> 
#include <iostream>
namespace Hex {
    template<class T>
    Sigmoid<T>::Sigmoid() {}

    template<class T>
    Sigmoid<T>::~Sigmoid() {
        output->cudafree(); input_error->cudafree();
    }
    
    template <typename T>
    __device__ T sigmoid(T x) {
        return static_cast<T>(1) / (static_cast<T>(1) + expf(-x));
    }
     
    template <typename T>
    __global__ void sigmoid_forward_kernel(const T* input, T* output, int batch_size, int feature_size) {
        int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (batch_idx < batch_size && feature_idx < feature_size) {
            int input_index = batch_idx * feature_size + feature_idx;
            output[input_index] = sigmoid(input[input_index]);
        }
    }
     
    template<class T>
    Tensor<T>& Sigmoid<T>::forward(Tensor<T>& input_tensor, bool Istraining) {
        input = input_tensor;
        output.reset(new Tensor<T>(input_tensor.getShape()));

        std::vector<int> shape = input.getShape();
        int batch_size = shape[0];
        int feature_size = shape[1];

        dim3 blockSize(16, 16); // You can adjust this block size as needed
        dim3 gridSize((batch_size + blockSize.x - 1) / blockSize.x, (feature_size + blockSize.y - 1) / blockSize.y);

        sigmoid_forward_kernel << <gridSize, blockSize >> > (input.getData(), output->getData(), batch_size, feature_size);

        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from sigmoid forward method : %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }

        return *output;
    }



    template <typename T>
    __device__ T sigmoid_derivative(T x) {
        T sigmoid_value = sigmoid(x);
        return sigmoid_value * (static_cast<T>(1) - sigmoid_value);
    }

    // Kernel for backward pass using sigmoid activation
    template <typename T>
    __global__ void sigmoid_backward_kernel(const T* input, const T* output_error, T* input_error, int batch_size, int feature_size) {
        int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (batch_idx < batch_size && feature_idx < feature_size) {
            int input_index = batch_idx * feature_size + feature_idx;
            input_error[input_index] = sigmoid_derivative(input[input_index]) * output_error[input_index];
        }
    }

    // Update backpropagation method with sigmoid computation
    template<class T>
    Tensor<T>& Sigmoid<T>::backpropagation(Tensor<T>& output_error, float learning_rate) {
        input_error.reset(new Tensor<T>(output_error.getShape()));

        std::vector<int> shape = output_error.getShape();
        int batch_size = shape[0];
        int feature_size = shape[1];

        dim3 blockSize(16, 16); // You can adjust this block size as needed
        dim3 gridSize((batch_size + blockSize.x - 1) / blockSize.x, (feature_size + blockSize.y - 1) / blockSize.y);

        sigmoid_backward_kernel << <gridSize, blockSize >> > (input.getData(), output_error.getData(), input_error->getData(), batch_size, feature_size);

        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from sigmoid backward method : %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }

        return *input_error;
    }

    template class Sigmoid<float>;
    template class Sigmoid<int>;
    template class Sigmoid<double>;
}