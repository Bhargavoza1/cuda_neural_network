#pragma once

#include <cuda_runtime.h>
#include <vector>
 
#include "Tensor.h"

#include "Errorhelper.cpp"

namespace Hex {
 
    template<class T, class U>
    __global__ void addKernel(const T* a, const U* b, typename std::common_type<T, U>::type* c, int size) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < size) {
            c[idx] = static_cast<typename std::common_type<T, U>::type>(a[idx] + b[idx]);
        }
    }

    template<class T, class U>
    std::unique_ptr<Tensor<typename std::common_type<T, U>::type>> addTensor(const Tensor<T>& tensor1, const Tensor<U>& tensor2) {

        if (tensor1.getShape() != tensor2.getShape()) {
            std::cerr << "Error: Tensor shapes must be the same for addition. Shape of tensor1: "
                << shapeToString(tensor1.getShape()) << ", Shape of tensor2: " << shapeToString(tensor2.getShape()) << std::endl;
            exit(EXIT_FAILURE); // or use exit(EXIT_FAILURE) if you prefer
        }


        using CommonType = typename std::common_type<T, U>::type;
        std::vector<int> shape = tensor1.getShape();
        
        std::unique_ptr<Tensor<CommonType>> result(new Tensor<CommonType>(shape));

        
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }

        CommonType* resultData;
        cudaMalloc((void**)&resultData, size * sizeof(CommonType));

        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        addKernel<<<gridSize, blockSize>>> (tensor1.getData(), tensor2.getData(), resultData , size);
        cudaDeviceSynchronize();

        // Update the data pointer in the result tensor
       
        result->setData(resultData );
 
        return result;
    }

    template<typename T>
    void initTensorOnGPU(Tensor<T>& tensor, float multiplier)
    {
        std::vector<int> shape = tensor.getShape();
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        T* d_data;
        cudaMalloc((void**)&d_data, size * sizeof(T));

        // Launch CUDA kernel to initialize and multiply the tensor
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        initializeTensor << <gridSize, blockSize >> > (d_data, size, multiplier);

        // Copy data back to the CPU if necessary
        cudaMemcpy(tensor.getData(), d_data, size * sizeof(T), cudaMemcpyDeviceToHost);

        // Free GPU memory
        cudaFree(d_data);
    }


 
   // CUDA kernel for tensor initialization with multiplication
    template <typename T>
    __global__ void initializeTensor(T* data, int size, float multiplier) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
           
            T value = static_cast<T>(index)  ;

            if (multiplier != 0) {
                value *= multiplier;
            }

            data[index] = value;
        }
    }

 
}
