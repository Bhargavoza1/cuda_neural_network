
#include "BatchNorm.h"
#include "../utils/tensor_oprations.h"
#include <cassert>
namespace Hex {

    template <class T>
    BatchNorm<T>::BatchNorm(int Batch_or_channels, TensorShape tensorshape , float momentum, float eps)
        : momentum(momentum), eps(eps) , _Tshape(tensorshape) ,
        gamma(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, Batch_or_channels, 1, 1 }) : Tensor<T>({ 1, Batch_or_channels })),
        beta(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, Batch_or_channels, 1, 1 }) : Tensor<T>({ 1, Batch_or_channels })),
        running_mean(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, Batch_or_channels, 1, 1 }) : Tensor<T>({ 1, Batch_or_channels })),
        running_var(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, Batch_or_channels, 1, 1 }) : Tensor<T>({ 1, Batch_or_channels })),
        input_mean(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, Batch_or_channels, 1, 1 }) : Tensor<T>({ 1, Batch_or_channels })),
        input_var(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, Batch_or_channels, 1, 1 }) : Tensor<T>({ 1, Batch_or_channels }))
    {
        initTensorToOneOnGPU(gamma);
        initTensorToOneOnGPU(running_var); 
    }

    template <class T>
    BatchNorm<T>::~BatchNorm() {
        // Destructor implementation
    }

    template <class T>
    Tensor<T>& BatchNorm<T>::forward(Tensor<T>& input_tensor, bool Istraining  ) {
        size_t tensor_dimensions = input_tensor.getShape().size();
        //std::cout << " size from batch norm forward : " << tensor_dimensions << std::endl;
        if (tensor_dimensions == 4 && _Tshape == TensorShape::_4D) {
            //gamma.print();
            return forward_4d(input_tensor, Istraining);
        }
        else if (tensor_dimensions == 2 && _Tshape == TensorShape::_2D) {
            return forward_2d(input_tensor, Istraining);
        }
        assert(false && "Invalid tensor dimensions or shape");
    }



    template<class T>
    Tensor<T>& BatchNorm<T>::forward_2d(Tensor<T>& input_tensor, bool Istraining)
    {
        return input_tensor;
    }
    template<class T>
    __global__ void batchnorm_forward4d_kernel(  const T* __restrict__ input_data,
        T* __restrict__ output_data,
        const T* __restrict__ gamma_data,
        const T* __restrict__ beta_data,
        T* __restrict__ running_mean,
        T* __restrict__ running_variance,
        T* __restrict__  x_normalized,
        T* __restrict__  input_mean,
        T* __restrict__  input_var,
        const int batch_size,
        const int out_channels,
        const int input_width,
        const int input_height,
        const float momentum,
        const float eps,
        const bool Istraining) {

        int batch_channel_idx = blockIdx.x;
        int batch_idx = batch_channel_idx / out_channels;
        int channel_idx = batch_channel_idx % out_channels;
        int output_row = blockIdx.y * blockDim.y + threadIdx.x;
        int output_col = blockIdx.z * blockDim.z + threadIdx.y;
 
        if (batch_idx < batch_size && channel_idx < out_channels && output_row < input_width && output_col < input_height) {
            int input_idx = batch_idx * out_channels * input_height * input_width + channel_idx * input_height * input_width + output_row * input_width + output_col;
            if (Istraining) {

                // Compute mean along height and width for each channel
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    T sum = 0;

                    for (int b = 0; b < batch_size; ++b) {
                        for (int i = 0; i < input_width; ++i) {
                            for (int j = 0; j < input_height; ++j) {
                                int data_idx = b * out_channels * input_height * input_width + channel_idx * input_height * input_width + i * input_height + j;
                                sum += input_data[data_idx];
 
                            }
                        }
                    }
                    input_mean[channel_idx] = sum / (batch_size * input_height * input_width);
                   
                    T diff = 0;
                    T sum_squares = 0.0f;
                    for (int b = 0; b < batch_size; ++b) {
                        for (int i = 0; i < input_width; ++i) {
                            for (int j = 0; j < input_height; ++j) {
                                int data_idx = b * out_channels * input_height * input_width + channel_idx * input_height * input_width + i * input_height + j;
                                diff = input_data[data_idx] - input_mean[channel_idx];
                                sum_squares += diff * diff;
                            }
                        }
                    }
                    input_var[channel_idx] = sum_squares / (batch_size * input_height * input_width);
 
                }

                __syncthreads();
           
                x_normalized[input_idx] = (input_data[input_idx] - input_mean[channel_idx]) / sqrtf(input_var[channel_idx] + eps);
                output_data[input_idx] = gamma_data[channel_idx] * x_normalized[input_idx] + beta_data[channel_idx];

                //self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                //    self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

                running_mean[channel_idx] = momentum * running_mean[channel_idx] + (1 - momentum) * input_mean[channel_idx]; 
                running_variance[channel_idx] = momentum * running_variance[channel_idx] + (1 - momentum) * input_var[channel_idx];

            }
            else {
                x_normalized[input_idx] = (input_data[input_idx] - running_mean[channel_idx]) / sqrtf(running_variance[channel_idx] + eps);
                output_data[input_idx] = gamma_data[channel_idx] * x_normalized[input_idx] + beta_data[channel_idx];
            }
        
        }
    }
 
    template<class T>
    Tensor<T>& BatchNorm<T>::forward_4d(Tensor<T>& input_tensor, bool Istraining)
    {
        input = input_tensor;
        x_normalized = input_tensor;
        output.reset(new Tensor<T>({ input.getShape()}));
       
        int  _batch_size = input.getShape()[0];
        int  _out_channels = input.getShape()[1];
        int  _in_width = input.getShape()[2];
        int  _in_height = input.getShape()[3];
 
  
        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks(_batch_size * _out_channels,
            (_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
 

        //// Compute mean and variance
        //Tensor<T> mean({ 1, _out_channels,1,1 });
        //Tensor<T> variance({ 1, _out_channels,1,1 });
    
 

        batchnorm_forward4d_kernel << <numBlocks, threadsPerBlock >> > (  input.getData(),
            output->getData(),
            gamma.getData(),
            beta.getData(),
            running_mean.getData(),
            running_var.getData(),
            x_normalized.getData(),
            input_mean.getData(),
            input_var.getData(),
            _batch_size,
            _out_channels,
            _in_width,
            _in_height,
            momentum,
            eps,
            Istraining); 
        cudaDeviceSynchronize();

        input_mean.print();
        input_var.print();
        x_normalized.print();
        output->print();
        running_mean.print();
        running_var.print();
        return *output;
    }

    template <class T>
    Tensor<T>& BatchNorm<T>::backpropagation(Tensor<T>& output_error, float learning_rate) {
        return output_error;
    }

    template class BatchNorm<float>;
    template class BatchNorm<int>;
    template class BatchNorm<double>;
}