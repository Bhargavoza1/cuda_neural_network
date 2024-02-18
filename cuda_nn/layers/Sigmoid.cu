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
    __device__ T sigmoid_derivative(T x) {
        T sigmoid_value = sigmoid(x);
        return sigmoid_value * (static_cast<T>(1) - sigmoid_value);
    }

    // Kernel for forward pass using sigmoid activation
    template <typename T>
    __global__ void sigmoid_forward_kernel(const T* input, T* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = sigmoid(input[idx]);
        }
    }

    // Kernel for backward pass using sigmoid activation
    template <typename T>
    __global__ void sigmoid_backward_kernel(const T* input, const T* output_error, T* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = sigmoid_derivative(input[idx]) * output_error[idx];
        }
    }

    // Update forward method with sigmoid computation
    template<class T>
    Tensor<T>& Sigmoid<T>::forward(Tensor<T>& input_tensor, bool Istraining) {
        input= input_tensor;
        output.reset(new Tensor<T>(input_tensor.getShape()));

        std::vector<int> shape = input.getShape();
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
      /*  std::cout << "dbug strat of sigmoid" << std::endl;
        std::cout << "intpu" << std::endl;
        input.print();
   */
        
        sigmoid_forward_kernel << <(size + 255) / 256, 256 >> > (input.getData(), output->getData(), size);
        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from sigmoid forward method : %s\n", cudaGetErrorString(cudaError));
             exit(EXIT_FAILURE);  // or handle the error appropriately
        }

        //std::cout << "output" << std::endl;
        //output->print();
        //std::cout << "dbug end of sigmoid" << std::endl;
        //std::cout <<   std::endl;
        //std::cout <<   std::endl;
        //std::cout <<   std::endl;
        return *output;
    }

    // Update backpropagation method with sigmoid computation
    template<class T>
    Tensor<T>& Sigmoid<T>::backpropagation(Tensor<T>& output_error, float learning_rate) {
        input_error.reset(new Tensor<T>(output_error.getShape()));

        std::vector<int> shape = output_error.getShape();
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        //std::cout << "back dbug strat of sigmoid" << std::endl;
        //std::cout << "intpu" << std::endl;
        //input.print();

        //std::cout << "back dbug end of sigmoid" << std::endl;
        sigmoid_backward_kernel << <(size + 255) / 256, 256 >> > (input.getData(), output_error.getData(), input_error->getData(), size);
        cudaDeviceSynchronize();
        //std::cout << "output" << std::endl;
        //output->print();
        //std::cout << "dbug end of relu" << std::endl;
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from sigmoid backword method : %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }
/*        std::cout << std::endl;
        std::cout << std::endl;*/   
        return *input_error;
    }
    // Explicit instantiation of the template class for supported types
    template class Sigmoid<float>;
    template class Sigmoid<int>;
    template class Sigmoid<double>;
}