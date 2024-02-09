 
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../utils/Tensor.h"
#include <memory>
 namespace Hex{
    template<typename T>
    __global__ void mse_mean_kernel(const T* y_true, const T* y_pred,  T* result, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        T local_sum = 0;

        if (idx < size) {
            T diff = y_true[idx] - y_pred[idx];
            local_sum += diff * diff;
             
        }

        // Calculate sum of squared differences using atomic add
        __shared__ T shared_sum;
        if (threadIdx.x == 0)
            shared_sum = 0;
        __syncthreads();

        atomicAdd(&shared_sum, local_sum);
        __syncthreads();

        if (idx == 0) {
            // Calculate mean
            *result = shared_sum / size;
        }
    }



    template<typename T>
    std::unique_ptr<Tensor<T>> mse(Tensor<T>& y_true, Tensor<T>& y_pred ) {
        std::unique_ptr<Tensor<T>> result(new Tensor<T>({1}));

        std::vector<int> shape = y_true.getShape();
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
       
        mse_mean_kernel << <(size + 255) / 256, 256 >> > (y_true.getData(), y_pred.getData(), result->getData(), size);
        cudaDeviceSynchronize(); 
        return result;
    }


    
    template<typename T>
    __global__ void mse_derivative_kernel(const T* y_true, const T* y_pred, T* derivative, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) { 
            derivative[idx] = (2.0 / static_cast<T>(size)) * (y_pred[idx] - y_true[idx]); 
        }
    }

    template<typename T>
    std::unique_ptr<Tensor<T>> mse_derivative(Tensor<T>& y_true, Tensor<T>& y_pred) {
        std::unique_ptr<Tensor<T>> derivative(new Tensor<T>(y_true.getShape()));

        std::vector<int> shape = y_true.getShape();
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
       
        mse_derivative_kernel << <(size + 255) / 256, 256 >> > (y_true.getData(), y_pred.getData(), derivative->getData(), size);
        cudaDeviceSynchronize(); 
        return derivative;
    }

 
  
     template std::unique_ptr<Tensor<float>> mse(Tensor<float>& y_true, Tensor<float>& y_pred);

   
     template std::unique_ptr<Tensor<float>> mse_derivative(Tensor<float>& y_true, Tensor<float>& y_pred);

          template std::unique_ptr<Tensor<int>> mse(Tensor<int>& y_true, Tensor<int>& y_pred);

   
     template std::unique_ptr<Tensor<int>> mse_derivative(Tensor<int>& y_true, Tensor<int>& y_pred);

          template std::unique_ptr<Tensor<double>> mse(Tensor<double>& y_true, Tensor<double>& y_pred);

   
     template std::unique_ptr<Tensor<double>> mse_derivative(Tensor<double>& y_true, Tensor<double>& y_pred);
}