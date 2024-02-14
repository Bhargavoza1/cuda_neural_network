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

        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        addKernel<<<gridSize, blockSize>>> (tensor1.getData(), tensor2.getData(), result->getData(), size);
        cudaDeviceSynchronize();

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("CUDA error from add tensor: %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }
        // Update the data pointer in the result tensor
       
        //result->setData(resultData);
 
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
        
        // Launch CUDA kernel to initialize and multiply the tensor
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        initializeTensor << <gridSize, blockSize >> > (tensor.getData(), size, multiplier);
        cudaDeviceSynchronize();

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("CUDA error from init: %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }
    }

#include <curand_kernel.h>
 
   // CUDA kernel for tensor initialization with multiplication
    template <typename T>
    __global__ void initializeTensor(T* data, int size, float multiplier) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            curandState state;
            curand_init(clock64(), index, 0, &state); // Initialize random number generator for each thread

            data[index] = curand_uniform(&state) * (2 * 127.f) - 127.f;
            //T value = static_cast<T>(index+1)  ;

            //if (multiplier != 0) {
            //    value *= multiplier;
            //}

            //data[index] = value;
        }
    }


    template <typename T>
    std::unique_ptr<Tensor<T>>  sliceFirstIndex(int firstIndex, const Tensor<T>& tensor) {
        if (firstIndex < 0 || firstIndex >= tensor.getShape()[0]) {
            throw std::out_of_range("Index out of range");
        }
        std::vector<int> shape = tensor.getShape();
        std::vector<int> newShape(shape.begin() + 1, shape.end()); // New shape without the first dimension
        std::unique_ptr<Tensor<T>> slicedTensor(new Tensor<T>(newShape));

        // Calculate offset for the first index
        int offset = firstIndex * shape[1] * shape[2]; // Assuming the shape is nxnxn

        // Copy data from original tensor to sliced tensor
        cudaMemcpy(slicedTensor->getData(), tensor.getData() + offset, newShape[0] * newShape[1] * sizeof(T), cudaMemcpyDeviceToDevice);

        return slicedTensor;
    }

    //template <typename T>
    //std::unique_ptr<Tensor<T>> slice(int index, Tensor<T> tensor)   {
    //    // Check if the index is within bounds
    //    if (index < 0 || index >= tensor.getShape()[0]) {
    //        throw std::out_of_range("Index out of bounds");
    //    }
    //    std::vector<int> shape = tensor.getShape();
    //    std::vector<int> sliced_shape(shape.begin() + 1, shape.end());
  
    //    std::unique_ptr<Tensor<T>> sliced_tensor(new Tensor<T>(sliced_shape));

    //    for (int i = 0; i < sliced_shape[0]; ++i) {
    //        std::vector<int> original_indices = { index, i };
    //        std::vector<int> sliced_indices = { i };
    //        for (size_t j = 1; j < shape.size(); ++j) {
    //            original_indices.push_back(0);
    //            sliced_indices.push_back(j - 1);
    //        }

    //        T value = tensor.get(original_indices);
    //        sliced_tensor->set(sliced_indices, value);
    //    }

    //    return sliced_tensor;
    //}
 

    template<typename T>
    __global__ void transpose_kernel(const T* input, T* output, int rows, int cols) {
        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (tid_x < cols && tid_y < rows) {
            output[tid_x * rows + tid_y] = input[tid_y * cols + tid_x];
        }
    }

    template <typename T>
    std::unique_ptr<Tensor<T>>  transpose(const Tensor<T>& tensor) {
        // Get the shape of the original tensor
        std::vector<int> original_shape = tensor.getShape();
      

        // Swap the dimensions
        std::vector<int> transposed_shape(original_shape.rbegin(), original_shape.rend());

        std::unique_ptr<Tensor<T>> transposed_tensor(new Tensor<T>(transposed_shape));

        dim3 threadsPerBlock(16, 16); // 16x16 threads per block
        dim3 numBlocks((original_shape[0] + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (original_shape[1] + threadsPerBlock.y - 1) / threadsPerBlock.y); // Adjust grid dimensions

        transpose_kernel<<<numBlocks, threadsPerBlock>>>(tensor.getData(), transposed_tensor->getData(), original_shape[0], original_shape[1]);

        return transposed_tensor;
    }
}
