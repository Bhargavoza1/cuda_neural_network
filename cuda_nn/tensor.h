// tensor.h

#pragma once

#include <vector>
#include <cuda_runtime.h>

namespace Hex{

	// CUDA kernel to initialize tensor data
	template <typename T>
	__global__ void initTensorKernel(T* data, size_t size) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < size) {
			data[tid] = static_cast<T>(tid);  
		}
	}

    template <typename T>
    class Tensor {
    private:
        static_assert(std::is_same<T, int>::value || std::is_same<T, float>::value || std::is_same<T, double>::value,
            "Tensor only supports int, float, and double types");
        std::vector<size_t> shape;
        T* data;
        bool isAllocatedOnGPU;

        void initializeGPUTensor(size_t size) {
            dim3 blockSize(256); // Adjust the block size based on your requirements
            dim3 gridSize(static_cast<unsigned int>((size + blockSize.x - 1) / blockSize.x));
            initTensorKernel << <gridSize, blockSize >> > (data, size);
            cudaDeviceSynchronize();
        }

        void initializeCPUTensor(size_t size) {
            for (size_t i = 0; i < size; ++i) {
                data[i] = static_cast<T>(i);   
            }
        }

    public:
        Tensor(const std::vector<size_t>& shape, bool allocateOnGPU = true)
            : shape(shape), isAllocatedOnGPU(allocateOnGPU) {
            size_t size = 1;
            for (size_t dim : shape) {
                size *= dim;
            }

            if (allocateOnGPU) {
                // Allocate memory on the GPU
                cudaMallocManaged(&data, size * sizeof(T));

                // Initialize GPU tensor data
                initializeGPUTensor(size);
            }
            else {
                // Allocate memory on the CPU
                data = new T[size];

                // Initialize CPU tensor data
                initializeCPUTensor(size);
            }
        }

        ~Tensor() {
            if (isAllocatedOnGPU) {
                // Deallocate memory on the GPU using cudaFree
                cudaFree(data);
            }
            else {
                // Deallocate memory on the CPU using delete[]
                delete[] data;
            }
        }

        T& operator()(const std::vector<size_t>& indices) {
            size_t flat_index = 0;
            size_t stride = 1;

            for (size_t i = 0; i < shape.size(); ++i) {
                flat_index += indices[i] * stride;
                stride *= shape[i];
            }

            return data[flat_index];
        }

 

        const std::vector<size_t>& getShape() const {
            return shape;
        }
        const T* getData() const {
            return data;
        }

        /**
        * @brief Copy data from the GPU to the host (CPU) memory.
        *
        * This function uses cudaMemcpy to transfer data from the specified GPU memory
        * location (src) to the specified host memory location (dst). The size parameter
        * determines the number of elements to copy, each of size sizeof(T) bytes.
        *
        * @param dst Pointer to the destination host memory.
        * @param src Pointer to the source GPU memory.
        * @param size Number of elements to copy.
        */
        void tohost(T* dst, const T* src, size_t size) {
            cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost);
        }


        /**
         * @brief Copy data from the host (CPU) to the GPU memory.
         *
         * This function uses cudaMemcpy to transfer data from the specified host memory
         * location (src) to the GPU memory location represented by the 'data' member.
         * The size parameter determines the number of elements to copy, each of size
         * sizeof(T) bytes.
         *
         * @param src Pointer to the source host memory.
         * @param size Number of elements to copy.
         */
        void toDevice(const T* src, size_t size) {
            cudaMemcpy(data, src, size * sizeof(T), cudaMemcpyHostToDevice);
        }
    };
}
