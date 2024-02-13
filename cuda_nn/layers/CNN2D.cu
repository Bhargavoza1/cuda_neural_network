#include "CNN2D.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
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
    void CNN2D<T>::init_weight_n_bias()
    {
        dim3 blockSize(16, 16); // Block size (16x16 threads per block)
        dim3 gridSize((_out_channels + blockSize.x - 1) / blockSize.x, (_in_channels * _kernel_size * _kernel_size + blockSize.y - 1) / blockSize.y); // Grid size

        cnn2d_W_B_init << <gridSize, blockSize >> > (weights.getData(), bias.getData(), _out_channels, _in_channels, _kernel_size, _w_b_range);
        cudaDeviceSynchronize();
        //weights.print();
        //bias.print();
    }

    template<class T>
    CNN2D<T>::CNN2D(const std::vector<int>& batch_width_height, const std::vector<int>& in_out_channels, int kernel_size, int padding, float w_b_range) :
        _batch_size(batch_width_height[0]), _in_channels(in_out_channels[0]), _out_channels(in_out_channels[1]), _kernel_size(kernel_size),
        _padding(padding), _w_b_range(w_b_range),
        weights(std::vector<int>{_out_channels, _in_channels, _kernel_size, _kernel_size  }),
        bias(std::vector<int>{_out_channels}),
        output(std::vector<int>{_batch_size, _out_channels, batch_width_height[2], batch_width_height[3] }),
        input(std::vector<int>{_batch_size, _in_channels, batch_width_height[2], batch_width_height[3] }),
        input_error(std::vector<int>{_batch_size, _in_channels, batch_width_height[2], batch_width_height[3]  })
    {
        init_weight_n_bias();
    }

    template<class T>
    CNN2D<T>::~CNN2D()
    {
       
    }
 

 
    template<class T>
    __global__ void convolutionforward(T* input, T* output, T* weight, T* bias,
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
                            int x_index = idx + j - 0;
                            int y_index = idy + i - 0; 
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
     
        input = input_tensor;
       
        int  _batch_size = input.getShape()[0];
        int  _in_width = input.getShape()[2];
        int  _in_height = input.getShape()[3];
       
        dim3 threadsPerBlock(16, 16, 1);
        dim3 numBlocks((_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (_batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

        convolutionforward << <numBlocks, threadsPerBlock >> > (input_tensor.getData(), output.getData(),
            weights.getData(), bias.getData(),
            _batch_size, _in_channels, _in_height, _in_width,
            _out_channels, _kernel_size, _padding);
        //output.print();
        return output; 
    }


 


    template<class T>
    __global__ void convolutionBackpropagationAndUpdate(
        T* weights, T* bias,
        const T* output_error, const T* input, T* input_error,
        float learning_rate,
        int batch_size, int out_channels, int in_channels, int kernel_size,
        int input_height, int input_width, int padding)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int idz = blockIdx.z * blockDim.z + threadIdx.z;

        if (idx < input_width && idy < input_height && idz < batch_size) {
            for (int c = 0; c < in_channels; ++c) {
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        T gradient_weight = 0;
                        for (int k = 0; k < out_channels; ++k) {
                            int x_index = idx - j + padding;
                            int y_index = idy - i + padding;
                            if (x_index >= 0 && x_index < input_width && y_index >= 0 && y_index < input_height) {
                                int output_index = idz * out_channels * input_height * input_width +
                                    k * input_height * input_width +
                                    y_index * input_width +
                                    x_index;
                                gradient_weight += output_error[output_index] * input[idz * in_channels * input_height * input_width + c * input_height * input_width + y_index * input_width + x_index];
                            }
                        }
                        int weight_index = idx + j + (idy + i) * input_width + c * kernel_size * kernel_size;
                        weights[weight_index] -= learning_rate * gradient_weight;
                    }
                }
            }
        }

        if (idx == 0 && idy == 0 && idz < out_channels) {
            T gradient_bias = 0;
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < input_height; ++i) {
                    for (int j = 0; j < input_width; ++j) {
                        int output_index = b * out_channels * input_height * input_width + idz * input_height * input_width + i * input_width + j;
                        gradient_bias += output_error[output_index];
                    }
                }
            }
            bias[idz] -= learning_rate * gradient_bias;
        }

        if (idx < input_width && idy < input_height && idz < batch_size) {
            for (int c = 0; c < in_channels; ++c) {
                T gradient_input = 0;
                for (int k = 0; k < out_channels; ++k) {
                    for (int i = 0; i < kernel_size; ++i) {
                        for (int j = 0; j < kernel_size; ++j) {
                            int x_index = idx - j + padding;
                            int y_index = idy - i + padding;
                            if (x_index >= 0 && x_index < input_width && y_index >= 0 && y_index < input_height) {
                                int weight_index = k * in_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + i * kernel_size + j;
                                int output_index = idz * out_channels * input_height * input_width +
                                    k * input_height * input_width +
                                    y_index * input_width +
                                    x_index;
                                gradient_input += output_error[output_index] * weights[weight_index];
                            }
                        }
                    }
                }
                int input_index = idz * in_channels * input_height * input_width + c * input_height * input_width + idy * input_width + idx;
                input_error[input_index] = gradient_input;
            }
        }
    }


    template<class T>
    Tensor<T>& CNN2D<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
    {
        int  _batch_size = output_error.getShape()[0];
        int  _in_width = output_error.getShape()[2];
        int  _in_height = output_error.getShape()[3];

        dim3 threadsPerBlock(16, 16, 1);
        dim3 numBlocks((_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (_batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);

        convolutionBackpropagationAndUpdate << <numBlocks, threadsPerBlock >> > (
            weights.getData(), bias.getData(),
            output_error.getData(), input.getData(), input_error.getData(),
            learning_rate,
            _batch_size, _out_channels, _in_channels, _kernel_size,
            _in_height, _in_width, _padding);
        cudaDeviceSynchronize();
        return input_error;
    }




    template class CNN2D<float>;
    template class CNN2D<int>;
    template class CNN2D<double>;
}