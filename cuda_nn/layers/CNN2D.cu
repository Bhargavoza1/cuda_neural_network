#include "CNN2D.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <memory>
//Output size = ((input size - kernel size + 2 * padding) / stride) + 1
//
// 
//
//input size = (Output size - 1) * stride + kernel size - 2 * padding

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
 
            weights[index] = static_cast<T>(index  );
        }

        if (row < out_channels && col == 0) {
            curandState state;
            curand_init(clock64(), row, 0, &state);  

            bias[row] = static_cast<T>(row +1 );  
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
    CNN2D<T>::CNN2D(const std::vector<int>& batch_width_height, const std::vector<int>& in_out_channels, int kernel_size, int padding,int stride,float w_b_range) :
        _batch_size(batch_width_height[0]), _in_channels(in_out_channels[0]), _out_channels(in_out_channels[1]), _kernel_size(kernel_size),
        _padding(padding), _stride(stride), _w_b_range(w_b_range),
        weights(std::vector<int>{_out_channels, _in_channels, _kernel_size, _kernel_size  }),
        bias(std::vector<int>{_out_channels})
       // output(std::vector<int>{_batch_size, _out_channels, batch_width_height[2], batch_width_height[3] }),
      //  input(std::vector<int>{_batch_size, _in_channels, batch_width_height[2], batch_width_height[3] }),
       // input_error(std::vector<int>{_batch_size, _in_channels, batch_width_height[2], batch_width_height[3]  })
    {
        init_weight_n_bias();
    }

    template<class T>
    CNN2D<T>::~CNN2D()
    {
        output->cudafree();
        input.cudafree();
        input_error->cudafree();
    }
 
  

    template<class T>
    __global__ void convolutionforward(const T* input, const T* weight, const T* bias, T* output,
        int batch_size, int in_channels, int in_width, int in_height,
        int out_channels, int kernel_size, int padding, int stride, int out_width, int out_height) {

        int batch_idx = blockIdx.x / out_channels;
        int channel_idx = blockIdx.x % out_channels;
        int output_row = blockIdx.y * blockDim.y + threadIdx.x;
        int output_col = blockIdx.z * blockDim.z + threadIdx.y;
      
        if (batch_idx < batch_size && channel_idx < out_channels && output_row < out_width && output_col < out_height) {
            int input_row_start = output_row * stride - padding;
            int input_col_start = output_col * stride - padding;
            T value = 0;

            for (int c = 0; c < in_channels; ++c) {
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        int input_row = input_row_start + i;
                        int input_col = input_col_start + j;
                        if (input_row >= 0 && input_row < in_height && input_col >= 0 && input_col < in_width) {
                            int input_idx = (batch_idx * in_channels * in_height * in_width) + (c * in_height * in_width) + (input_row * in_width) + input_col;
                            int weight_idx = (channel_idx * in_channels * kernel_size * kernel_size) + (c * kernel_size * kernel_size) + (i * kernel_size) + j;
                          
                            value += input[input_idx] * weight[weight_idx];
                          //  printf("(%dx%dx%dx%d  \n", batch_idx , c, output_row, output_col);
                        }
                    }
                }
            }
           // printf("(%f  \n", value);
            int output_idx = (batch_idx * out_channels * out_height * out_width) + (channel_idx * out_height * out_width) + (output_row * out_width) + output_col;
            output[output_idx] = value  ; 
            //printf("(%dx%dx%dx%d)  \n", batch_idx ,  channel_idx  , output_row , output_col);
        }
    }





    template<class T>
    Tensor<T>& CNN2D<T>::forward(Tensor<T>& input_tensor, bool Istraining)
    {
     
        input = input_tensor;

    
        int  _batch_size = input.getShape()[0];
        int  _in_width = input.getShape()[2];
        int  _in_height = input.getShape()[3];
      
        int _out_width = ((_in_width - _kernel_size + 2 * _padding) / _stride) + 1;
        int _out_height = ((_in_height - _kernel_size + 2 * _padding) / _stride) + 1;

        //std::cout << _in_width << _in_height << _out_width << _out_height;
        output.reset(new Tensor<T>({ _batch_size , _out_channels ,_out_width , _out_height }));
        //input.print();
        dim3 threadsPerBlock(8,8,8);
        dim3 numBlocks(_batch_size * _out_channels ,
            (_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
 
        convolutionforward << <numBlocks, threadsPerBlock >> > (input_tensor.getData(), 
            weights.getData(), bias.getData(), output->getData(),
            _batch_size, _in_channels, _in_width, _in_height,
            _out_channels, _kernel_size, _padding, _stride, _out_width , _out_height);
        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from liner backword method : %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }
        //std::cout << "weights" << std::endl;
        // weights.print();
      //  output->print();
        return *output; 
    }


    template<class T>
    __global__ void convolutionBackwardInputError(const T* output_error, const T* input, T* weight, T* bias, T* input_error,
        int batch_size, int in_channels, int in_width, int in_height,
        int out_channels, int kernel_size, int padding, int stride, int out_width, int out_height , float learning_rate) {

        int batch_idx = blockIdx.x / in_channels;
        int channel_idx = blockIdx.x % in_channels;
        int input_row = blockIdx.y * blockDim.y + threadIdx.x;
        int input_col = blockIdx.z * blockDim.z + threadIdx.y;

        if (batch_idx < batch_size && channel_idx < in_channels && input_row < in_height && input_col < in_width) {
            T value = 0;
           
            int input_idx = (batch_idx * in_channels * in_height * in_width) + (channel_idx * in_height * in_width) + (input_row * in_width) + input_col;
            for (int c = 0; c < out_channels; ++c) {
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        int output_row = (input_row + padding - i) / stride;
                        int output_col = (input_col + padding - j) / stride;
                        if (output_row >= 0 && output_row < out_height && output_col >= 0 && output_col < out_width) {
                            int output_idx = (batch_idx * out_channels * out_height * out_width) + (c * out_height * out_width) + (output_row * out_width) + output_col;
                            int weight_idx = (c * in_channels * kernel_size * kernel_size) + (channel_idx * kernel_size * kernel_size) + (i * kernel_size) + j;
                            value += output_error[output_idx] * weight[weight_idx];


                            ////////////////////// Update weight only if within bounds
                            // printf("before weight[weight_idx] %d = %f \n", weight_idx, weight[weight_idx]);
                            //if (input_row - padding + i * stride >= 0 && input_col - padding + j * stride >= 0) {
                            atomicAdd(&weight[weight_idx], -learning_rate * output_error[output_idx] * input[input_idx]);
                            // }
                             // printf("after weight[weight_idx] %d = %f \n", weight_idx , weight[weight_idx]);
                            // No need to update bias here since it's handled outside the loop
                        }
                    }
                }
            }
           // int input_idx = (batch_idx * in_channels * in_height * in_width) + (channel_idx * in_height * in_width) + (input_row * in_width) + input_col;
            input_error[input_idx] = value;
        }
 
        // Update bias
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            for (int c = 0; c < out_channels; ++c) { 
                 //printf("before bias[c] %d =  %f \n",c , bias[c]);
                atomicAdd(&bias[c], -learning_rate * output_error[(batch_idx * out_channels * out_height * out_width) + (c * out_height * out_width)]);
                 // printf("after bias[c]  %d =  %f \n", c, bias[c]);
            }
        }
    }

 


    template<class T>
    Tensor<T>& CNN2D<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
    {
        int  _batch_size = input.getShape()[0];
       // int  _in_width = input.getShape()[2];
       // int  _in_height = input.getShape()[3];
        int _out_width = output_error.getShape()[2];
        int _out_height = output_error.getShape()[3];
        int _in_width = (_out_width - 1) * _stride - 2 * _padding + _kernel_size;
        int _in_height = (_out_height - 1) * _stride - 2 * _padding + _kernel_size; 
       // output_error.print();

        input_error.reset(new Tensor<T>({ _batch_size , _in_channels ,_in_width , _in_height }));
        dim3 threadsPerBlock(8 ,8, 8);
        dim3 numBlocks(_batch_size * _out_channels,
            (_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y
        );
 
        
        convolutionBackwardInputError << <numBlocks, threadsPerBlock >> > ( output_error.getData(), input.getData() ,
            weights.getData(), bias.getData() , input_error->getData(), _batch_size, _in_channels, _in_width, _in_height,
            _out_channels, _kernel_size, _padding, _stride, _out_width, _out_height, learning_rate);

        cudaDeviceSynchronize();
        //cudaError_t cudaError = cudaGetLastError();
        //if (cudaError != cudaSuccess) {
        //    printf("error from liner backword method : %s\n", cudaGetErrorString(cudaError));
        //    exit(EXIT_FAILURE);  // or handle the error appropriately
        //}
        return *input_error;
    }




    template class CNN2D<float>;
    template class CNN2D<int>;
    template class CNN2D<double>;
}