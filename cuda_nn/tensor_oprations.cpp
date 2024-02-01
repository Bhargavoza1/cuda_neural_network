#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
 
#include "Tensor.h"

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
}
