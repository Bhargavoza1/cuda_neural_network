#include "CNN2D.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
namespace Hex
{
    template<class T>
    __global__ void cnn2d_W_B_init(T* weights, T* bias, int out_channels, int in_channels, int kernel_size, float w_b_range) {
        //int row = blockIdx.y * blockDim.y + threadIdx.y;
        //int col = blockIdx.x * blockDim.x + threadIdx.x;

        //if (row < out_channels && col < in_channels * kernel_size * kernel_size) {
        //    int index = row * (in_channels * kernel_size * kernel_size) + col;
        //    curandState state;
        //    curand_init(clock64(), index, 0, &state); // Initialize random number generator for each thread

        //    weights[index] = curand_uniform(&state) * (2 * w_b_range) - w_b_range; // Generate random number in range [-w_b_range, w_b_range]
        //}

        //if (row < out_channels && col == 0) {
        //    curandState state;
        //    curand_init(clock64(), row, 0, &state); // Initialize random number generator for each thread

        //    bias[row] = curand_uniform(&state) * (2 * w_b_range) - w_b_range; // Generate random number in range [-w_b_range, w_b_range]
        //}


        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < out_channels && col < in_channels * kernel_size * kernel_size) {
            int index = row * (in_channels * kernel_size * kernel_size) + col;
 
            weights[index] = static_cast<T>(index + 1);
        }

        if (row < out_channels && col == 0) {
            curandState state;
            curand_init(clock64(), row, 0, &state);  

            bias[row] = static_cast<T>(row + 1);  
        }

        
    }

    template<class T>
    CNN2D<T>::CNN2D(const std::vector<int>& batch_height_width, const std::vector<int>& in_out_channels, int kernel_size, int padding, float w_b_range) :
        _batch_size(batch_height_width[0]), _in_channels(in_out_channels[0]), _out_channels(in_out_channels[1]), _kernel_size(kernel_size),
        _padding(padding), _w_b_range(w_b_range),
        weights(std::vector<int>{_out_channels, _in_channels, _kernel_size, _kernel_size  }),
        bias(std::vector<int>{_out_channels}),
        output(std::vector<int>{_batch_size, _out_channels, batch_height_width[1], batch_height_width[2] }),
        input(std::vector<int>{_batch_size, _in_channels, batch_height_width[1], batch_height_width[2] }),
        input_error(std::vector<int>{_batch_size, _in_channels, batch_height_width[1], batch_height_width[2]  })
    {

    }

    template<class T>
    CNN2D<T>::~CNN2D()
    {
        init_weight_n_bias();
    }

    //template<class T>
    //CNN2D<T>::CNN2D(int a)
    //{
    //    dim3 blockSize(2, 2); // 2x2 thread block
    //    dim3 gridSize(2, 2);   // 2x2 grid
    //    float b = 2.0f;
    //    cnnweight << <gridSize, blockSize >> > (a, b);
    //    cudaDeviceSynchronize();
    //    cudaError_t cudaError = cudaGetLastError();
    //    if (cudaError != cudaSuccess) {
    //        printf("error from liner backword method : %s\n", cudaGetErrorString(cudaError));
    //        exit(EXIT_FAILURE);  // or handle the error appropriately
    //    }
    //}


#define MAX_KERNEL_SIZE 5

    __global__ void convolutionforward(float* input, float* output, float* weight, float* bias,
        int batch_size, int in_channels, int in_height, int in_width,
        int out_channels, int kernel_size, int padding) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int idz = blockIdx.z * blockDim.z + threadIdx.z;

        if (idx < in_width && idy < in_height && idz < batch_size) {
            for (int k = 0; k < out_channels; ++k) {
                float sum = 0.0f;
                for (int c = 0; c < in_channels; ++c) {
                    for (int i = 0; i < kernel_size; ++i) {
                        for (int j = 0; j < kernel_size; ++j) {
                            int x_index = idx + j - padding;
                            int y_index = idy + i - padding;
                            if (x_index >= 0 && x_index < in_width && y_index >= 0 && y_index < in_height) {
                                int input_index = idz * in_channels * in_height * in_width +
                                    c * in_height * in_width +
                                    y_index * in_width +
                                    x_index;
                                int weight_index = k * in_channels * kernel_size * kernel_size +
                                    c * kernel_size * kernel_size +
                                    i * kernel_size +
                                    j;
                                sum += input[input_index] * weight[weight_index];
                            }
                        }
                    }
                }
                int output_index = idz * out_channels * in_height * in_width +
                    k * in_height * in_width +
                    idy * in_width +
                    idx;
                output[output_index] = sum + bias[k];
            }
        }
    }


    template<class T>
    Tensor<T>& CNN2D<T>::forward(Tensor<T>& input_tensor)
    {
        return input_tensor;
    }

    template<class T>
    Tensor<T>& CNN2D<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
    {
        return output_error;
    }

    template<class T>
    void CNN2D<T>::init_weight_n_bias()
    {
        dim3 blockSize(16, 16); // Block size (16x16 threads per block)
        dim3 gridSize((_out_channels + blockSize.x - 1) / blockSize.x, (_in_channels * _kernel_size * _kernel_size + blockSize.y - 1) / blockSize.y); // Grid size
       
        cnn2d_W_B_init<<<gridSize, blockSize>>>(weights.getData(), bias.getData(), _out_channels, _in_channels, _kernel_size, _w_b_range);
        cudaDeviceSynchronize();
        weights.print();
        bias.print();
    }

    template class CNN2D<float>;
    template class CNN2D<int>;
    template class CNN2D<double>;
}