
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
        running_var(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, Batch_or_channels, 1, 1 }) : Tensor<T>({ 1, Batch_or_channels }))
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
        T* __restrict__ mean,
        T* __restrict__ variance,
        const int batch_size,
        const int out_channels,
        const int input_width,
        const int input_height,
        const float eps,
        const bool Istraining) {

        int batch_channel_idx = blockIdx.x;
        int batch_idx = batch_channel_idx / out_channels;
        int channel_idx = batch_channel_idx % out_channels;
        int output_row = blockIdx.y * blockDim.y + threadIdx.x;
        int output_col = blockIdx.z * blockDim.z + threadIdx.y;

        __shared__ T localmean[MAX_CHANNELS] ;
        __shared__ T localvariance[MAX_CHANNELS] ;

        if (batch_idx < batch_size && channel_idx < out_channels && output_row < input_width && output_col < input_height) {

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
                localmean[channel_idx] = sum / (batch_size * input_height * input_width);
                mean[channel_idx] = localmean[channel_idx];


                T diff = 0;
                T sum_squares = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    for (int i = 0; i < input_width; ++i) {
                        for (int j = 0; j < input_height; ++j) {
                            int data_idx = b * out_channels * input_height * input_width + channel_idx * input_height * input_width + i * input_height + j;
                            diff = input_data[data_idx] - localmean[channel_idx];
                            sum_squares += diff * diff;
                        }
                    }
                }
                localvariance[channel_idx] = sum_squares / (batch_size * input_height * input_width);
                variance[channel_idx] = localvariance[channel_idx];
            }

            __syncthreads();

            int input_idx = batch_idx * out_channels * input_height * input_width + channel_idx * input_height * input_width + output_row * input_width + output_col;
            output_data[input_idx] = (input_data[input_idx] - localmean[channel_idx]) / sqrtf(localvariance[channel_idx] + eps);
        
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
            _batch_size,
            _out_channels,
            _in_width,
            _in_height,
            eps,
           Istraining); 

        running_mean.print();
        running_var.print();
        output->print();
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