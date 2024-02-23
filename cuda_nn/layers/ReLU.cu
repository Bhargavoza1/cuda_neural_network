#include "ReLU.h"
#include <cuda_runtime.h>
namespace Hex {
    template<class T>
    ReLU<T>::ReLU()
    {
    }
    template<class T>
    ReLU<T>::~ReLU()
    {
        output->cudafree();
        input_error->cudafree();
    }

    template <typename T>
    __global__ void relu_forward_kernel(const T* input, T* output, int batch_size, int feature_size) {
        int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (batch_idx < batch_size && feature_idx < feature_size) {
            int input_index = batch_idx * feature_size + feature_idx;
            output[input_index] = max(input[input_index], static_cast<T>(0));
        }
    }

    template<class T>
    Tensor<T>& ReLU<T>::forward(Tensor<T>& input_tensor, bool Istraining)
    {
        input = input_tensor;
        output.reset(new Tensor<T>(input_tensor.getShape()));

        std::vector<int> shape = input.getShape();
        int batch_size = shape[0];
        int feature_size = shape[1];

        dim3 blockSize(16, 16);  
        dim3 gridSize((batch_size + blockSize.x - 1) / blockSize.x, (feature_size + blockSize.y - 1) / blockSize.y);

        relu_forward_kernel << <gridSize, blockSize >> > (input.getData(), output->getData(), batch_size, feature_size);

        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from relu forward method : %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);   
        }

        return *output;
    }

     
    template <typename T>
    __global__ void relu_backward_kernel(const T* input, const T* output_error, T* input_error, int batch_size, int feature_size) {
        int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (batch_idx < batch_size && feature_idx < feature_size) {
            int input_index = batch_idx * feature_size + feature_idx;
            input_error[input_index] = (input[input_index] > static_cast<T>(0)) ? output_error[input_index] : static_cast<T>(0);
        }
    }

    template<class T>
    Tensor<T>& ReLU<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
    {
        input_error.reset(new Tensor<T>(output_error.getShape()));

        std::vector<int> shape = output_error.getShape();
        int batch_size = shape[0];
        int feature_size = shape[1];

        dim3 blockSize(16, 16); // You can adjust this block size as needed
        dim3 gridSize((batch_size + blockSize.x - 1) / blockSize.x, (feature_size + blockSize.y - 1) / blockSize.y);

        relu_backward_kernel << <gridSize, blockSize >> > (input.getData(), output_error.getData(), input_error->getData(), batch_size, feature_size);

        cudaDeviceSynchronize();

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from relu backword method : %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }

        return *input_error;
    }


    // Explicit instantiation of the template class for supported types
    template class ReLU<float>;
    template class ReLU<int>;
    template class ReLU<double>;
}