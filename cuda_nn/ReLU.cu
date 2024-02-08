#include "ReLU.h"
#include "tensor_oprations.h"
namespace Hex {
    template<class T>
    ReLU<T>::ReLU()
    {
    }
    template<class T>
    ReLU<T>::~ReLU()
    {
    }

    template <typename T>
    __global__ void relu_forward_kernel(const T* input, T* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = max(input[idx], static_cast<T>(0));
        }
    }

    template<class T>
    Tensor<T>& ReLU<T>::forward(Tensor<T>& input_tensor)
    {
        input =  input_tensor ;
        output.reset(new Tensor<T>(input_tensor.getShape()));
        
        //std::cout << "dbug strat of relu" << std::endl;
        //std::cout << "intpu" << std::endl;
        //input.print();
 

        std::vector<int> shape = input.getShape();
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
 
        relu_forward_kernel << <(size + 255) / 256, 256 >> > (input.getData(), output->getData(), size);
        //std::cout << "output" << std::endl;
        //output->print();
        //std::cout << "dbug end of relu" << std::endl;
        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from relu forward method : %s\n", cudaGetErrorString(cudaError));
             exit(EXIT_FAILURE);  // or handle the error appropriately
        }
        //std::cout << std::endl;
        //std::cout << std::endl;
        return *output;
    }

     
    template <typename T>
    __global__ void relu_backward_kernel(const T* input, const T* output_error, T* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            // If the input value is greater than 0, set the gradient to 1, otherwise set it to 0
            output[idx] = (input[idx] > static_cast<T>(0)) ? output_error[idx] : static_cast<T>(0)   ;
        }
    }
    template<class T>
    Tensor<T>& ReLU<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
    {
        // Create a new tensor to hold the gradients
        input_error.reset(new Tensor<T>(output_error.getShape()));
   /*     std::cout << "dbug strat of relu backword" << std::endl;
        std::cout << "intpu" << std::endl;
        input.print();
 
        std::cout << "dbug end of relu" << std::endl;*/
     
  
        std::vector<int> shape = output_error.getShape();
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }

        // Call CUDA kernel to compute gradients
        relu_backward_kernel << <(size + 255) / 256, 256 >> > (input.getData(), output_error.getData(), input_error->getData(), size);

        // Synchronize to make sure the kernel has finished
        cudaDeviceSynchronize();

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from relu backword method : %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }
        //std::cout << std::endl;
        //std::cout << std::endl;
        return *input_error;
    }

    // Explicit instantiation of the template class for supported types
    template class ReLU<float>;
    template class ReLU<int>;
    template class ReLU<double>;
}