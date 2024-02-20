
#include "BatchNorm.h"
#include "../utils/tensor_oprations.h"
#include <cassert>
namespace Hex {

    template <class T>
    BatchNorm<T>::BatchNorm(int features_or_channels, TensorShape tensorshape , float momentum, float eps)
        : momentum(momentum), eps(eps) , _Tshape(tensorshape) ,
        gamma(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, features_or_channels, 1, 1 }) : Tensor<T>({ features_or_channels, 1 })),
        beta(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, features_or_channels, 1, 1 }) : Tensor<T>({ features_or_channels, 1 })),
        running_mean(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, features_or_channels, 1, 1 }) : Tensor<T>({ features_or_channels, 1 })),
        running_var(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, features_or_channels, 1, 1 }) : Tensor<T>({ features_or_channels, 1 })),
        input_mean(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, features_or_channels, 1, 1 }) : Tensor<T>({ features_or_channels, 1 })),
        input_var(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, features_or_channels, 1, 1 }) : Tensor<T>({ features_or_channels, 1 }))
    {
        initTensorToOneOnGPU(gamma);
        initTensorToOneOnGPU(running_var); 
    }

    template <class T>
    BatchNorm<T>::~BatchNorm() {
        // Destructor implementation
        delete x_normalized; 
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
    __global__ void batchnorm_forward_2d_kernel(const T* __restrict__ input_data,
        T* __restrict__ output_data,
        const T* __restrict__ gamma_data,
        const T* __restrict__ beta_data,
        T* __restrict__ running_mean,
        T* __restrict__ running_variance,
        T* __restrict__ x_normalized,
        T* __restrict__ input_mean,
        T* __restrict__ input_var,
        const int features,
        const int batch_size,
        const float momentum,
        const float eps,
        const bool Istraining) {

        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < features && col < batch_size) {
            int input_idx = row * batch_size + col;

            if (Istraining) {
                // Calculate mean
                if (threadIdx.y == 0) {
                    T sum = 0;
                    for (int b = 0; b < batch_size; ++b) {
                        int data_idx = row * batch_size + b;
                        sum += input_data[data_idx];
                    }
                    input_mean[row] = sum / (batch_size);

                    T diff = 0;
                    T sum_squares = 0.0f;
                    for (int b = 0; b < batch_size; ++b) {
                        int data_idx = row * batch_size + b;
                        diff = input_data[data_idx] - input_mean[row];
                        sum_squares += diff * diff;
                    }
                    input_var[row] = sum_squares / (batch_size);
                }
                __syncthreads();
 
                x_normalized[input_idx] = (input_data[input_idx] - input_mean[row]) / sqrtf(input_var[row] + eps);
                output_data[input_idx] = gamma_data[row] * x_normalized[input_idx] + beta_data[row];

                running_mean[row] = momentum * running_mean[row] + (1 - momentum) * input_mean[row];
                running_variance[row] = momentum * running_variance[row] + (1 - momentum) * input_var[row];
 
            }
            else {
                x_normalized[input_idx] = (input_data[input_idx] - running_mean[row]) / sqrtf(running_variance[row] + eps);
                output_data[input_idx] = gamma_data[row] * x_normalized[input_idx] + beta_data[row];
            }
        }
    }

    template<class T>
    Tensor<T>& BatchNorm<T>::forward_2d(Tensor<T>& input_tensor, bool Istraining)
    {

        input = input_tensor;
       
        x_normalized = new Tensor<T>({ input_tensor.getShape() });
        output.reset(new Tensor<T>({ input_tensor.getShape() }));

        int _fetures = input.getShape()[0];
        int _batch_size = input.getShape()[1];
 
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks( (_fetures + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
       // input.print();
        batchnorm_forward_2d_kernel << < numBlocks, threadsPerBlock >> > (input.getData(),
            output->getData(),
            gamma.getData(),
            beta.getData(),
            running_mean.getData(),
            running_var.getData(),
            x_normalized->getData(),
            input_mean.getData(),
            input_var.getData(),
            _fetures,
            _batch_size,
            momentum,
            eps,
            Istraining);
        cudaDeviceSynchronize(); 
        //input.print();
         //input_mean.print();
         //input_var.print();
         //x_normalized.print();
         //output->print();
         //running_mean.print();
         //running_var.print();
         //x_normalized->print();
        return *output;
    }

    template<class T>
    __global__ void batchnorm_forward_4d_kernel(  const T* __restrict__ input_data,
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
        x_normalized = new Tensor<T>({ input_tensor.getShape() });
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
    
 

        batchnorm_forward_4d_kernel << <numBlocks, threadsPerBlock >> > (  input.getData(),
            output->getData(),
            gamma.getData(),
            beta.getData(),
            running_mean.getData(),
            running_var.getData(),
            x_normalized->getData(),
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

        //input_mean.print();
        //input_var.print();
        //x_normalized.print();
        //output->print();
        //running_mean.print();
        //running_var.print();
        return *output;
    }



    template <class T>
    Tensor<T>& BatchNorm<T>::backpropagation(Tensor<T>& output_error, float learning_rate) {

        size_t tensor_dimensions = output_error.getShape().size();
        //std::cout << " size from batch norm forward : " << tensor_dimensions << std::endl;
        if (tensor_dimensions == 4 && _Tshape == TensorShape::_4D) {
            //gamma.print();
            return backpropagation_4d(output_error, learning_rate);
        }
        else if (tensor_dimensions == 2 && _Tshape == TensorShape::_2D) {
            return backpropagation_2d(output_error, learning_rate);
        }
        assert(false && "Invalid tensor dimensions or shape");
      
    }


    template<class T>
    __global__ void batchnorm_backward_2d_kernel(const T* __restrict__ input_data,
        const T* __restrict__ output_error, 
        const T* __restrict__ x_normalized,
        const T* __restrict__ input_mean,
        const T* __restrict__ input_var, 
        T* __restrict__ input_error,
        T* __restrict__ gamma_gradient,
        T* __restrict__ beta_gradient,
        T* __restrict__ grad_normalized ,
        const int features,
        const int batch_size,
        const float eps)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < features && col < batch_size) {
            int input_idx = row * batch_size + col;
            T grad_gamma = 0.0;
            T grad_beta = 0.0; 
            
            // Calculate gradient of beta and gamma
 
         
            grad_normalized[input_idx] = output_error[input_idx] * gamma_gradient[row];
           //  printf("dmean %f \n", grad_normalized[input_idx]);
            // Calculate dvar
        
            T dvar = 0.0f;
            if (threadIdx.y == 0) {
                
                for (int b = 0; b < batch_size; ++b) {
                    int data_idx = row * batch_size + b;
                    T r = (input_data[data_idx] - input_mean[row]);
                    T t = pow(input_var[row] + eps, -1.5);
                    dvar += grad_normalized[data_idx] * r * -0.5 * t ;
                    
                }
               
            }
            __syncthreads(); 

            // Calculate dmean

            T dmean =   0.0f;
            if (threadIdx.y == 0) {

                T a = 0.0;
                T d = 0.0;
                for (int b = 0; b < batch_size; ++b) {
                    int data_idx = row * batch_size + b;
                    a += grad_normalized[data_idx] * (-1 / sqrt(input_var[row] + eps));
                    d += (-2 * (input_data[data_idx] - input_mean[row])) / batch_size;
                }
                dmean = a* dvar + d;
            }
            __syncthreads();

            for (int b = 0; b < batch_size; ++b) {
                int data_idx = row * batch_size + b;
               
                input_error[data_idx] = grad_normalized[data_idx] / sqrt(input_var[row] + eps) + dvar * 2.0 * (input_data[data_idx] - input_mean[row]) / batch_size + dmean / batch_size;
                
            }

            if (threadIdx.y == 0) {
            

                for (int b = 0; b < batch_size; ++b) {
                    int data_idx = row * batch_size + b;
                    grad_gamma += output_error[data_idx] * x_normalized[data_idx];
                    grad_beta += output_error[data_idx];
                }

                gamma_gradient[row] = grad_gamma;
                beta_gradient[row] = grad_beta;
            }
            __syncthreads();

        }
    }


    template<class T>
    Tensor<T>& BatchNorm<T>::backpropagation_2d(Tensor<T>& output_error, float learning_rate)
    {
        input_error.reset(new Tensor<T>({ input.getShape() }));
        grad_normalized.reset(new Tensor<T>({ input.getShape() }));
        const int features = input.getShape()[0];
        const int batch_size = input.getShape()[1];
       // input.print();
        const dim3 blockSize(16, 16); // Adjust block size as needed
        const dim3 gridSize((features + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y); // Adjust grid size as needed
       // input_error->print();
        // Invoke the CUDA kernel for backpropagation
        batchnorm_backward_2d_kernel << <gridSize, blockSize >> > (input.getData(),
            output_error.getData(), 
            x_normalized->getData(),
            input_mean.getData(),
            input_var.getData(), 
            input_error->getData(),
            gamma.getData(),
            beta.getData(),
            grad_normalized->getData(),
            features,
            batch_size,
            eps );

        // Synchronize to ensure the kernel is finished
        cudaDeviceSynchronize();
        //std::cout << " aaaaaaaaaaaaaaa" << std::endl;
        // gamma.print();
        // beta.print();
        return *input_error;
    }


    template<class T>
    __global__ void batchnorm_backward_4d_kernel(const T* __restrict__ input_data,
        const T* __restrict__ output_error,
        const T* __restrict__ x_normalized,
        const T* __restrict__ input_mean,
        const T* __restrict__ input_var,
        T* __restrict__ input_error,
        T* __restrict__ gamma_gradient,
        T* __restrict__ beta_gradient,
        T* __restrict__ grad_normalized,
        const int batch_size,
        const int out_channels,
        const int input_width,
        const int input_height,
        const float eps)
    {
        int batch_channel_idx = blockIdx.x;
        int batch_idx = batch_channel_idx / out_channels;
        int channel_idx = batch_channel_idx % out_channels;
        int output_row = blockIdx.y * blockDim.y + threadIdx.x;
        int output_col = blockIdx.z * blockDim.z + threadIdx.y;

        if (batch_idx < batch_size && channel_idx < out_channels && output_row < input_width && output_col < input_height) {
        }

            int input_idx = batch_idx * out_channels * input_height * input_width + channel_idx * input_height * input_width + output_row * input_width + output_col;

            grad_normalized[input_idx] = output_error[input_idx] * gamma_gradient[channel_idx];


            //if (threadIdx.x == 0 && threadIdx.y == 0) {
            //    T sum = 0;

            //    for (int b = 0; b < batch_size; ++b) {
            //        for (int i = 0; i < input_width; ++i) {
            //            for (int j = 0; j < input_height; ++j) {
            //                int data_idx = b * out_channels * input_height * input_width + channel_idx * input_height * input_width + i * input_height + j;
            //                sum += input_data[data_idx];

            //            }
            //        }
            //    }
            //    input_mean[channel_idx] = sum / (batch_size * input_height * input_width);

 

            //}

            //__syncthreads();

        //if (x < _in_width && y < _in_height) {
        //    int input_idx = ((b * _out_channels + oc) * _in_width + x) * _in_height + y;
        //    T grad_gamma = 0.0;
        //    T grad_beta = 0.0;
        //    T dvar = 0.0f;
        //    T dmean = 0.0f;

        //    grad_normalized[input_idx] = output_error[input_idx] * gamma_gradient[oc];

        //    if (threadIdx.y == 0) {
        //        for (int b = 0; b < _batch_size; ++b) {
        //            int data_idx = ((b * _out_channels + oc) * _in_width + x) * _in_height + y;
        //            T r = (input_data[data_idx] - input_mean[oc]);
        //            T t = pow(input_var[oc] + eps, -1.5);
        //            dvar += grad_normalized[data_idx] * r * -0.5 * t;
        //        }
        //    }
        //    __syncthreads();

        //    if (threadIdx.y == 0) {
        //        T a = 0.0;
        //        T d = 0.0;
        //        for (int b = 0; b < _batch_size; ++b) {
        //            int data_idx = ((b * _out_channels + oc) * _in_width + x) * _in_height + y;
        //            a += grad_normalized[data_idx] * (-1 / sqrt(input_var[oc] + eps));
        //            d += (-2 * (input_data[data_idx] - input_mean[oc])) / _batch_size;
        //        }
        //        dmean = a * dvar + d;
        //    }
        //    __syncthreads();

        //    for (int b = 0; b < _batch_size; ++b) {
        //        int data_idx = ((b * _out_channels + oc) * _in_width + x) * _in_height + y;
        //        input_error[data_idx] = grad_normalized[data_idx] / sqrt(input_var[oc] + eps) + dvar * 2.0 * (input_data[data_idx] - input_mean[oc]) / _batch_size + dmean / _batch_size;
        //    }

        //    if (threadIdx.y == 0) {
        //        for (int b = 0; b < _batch_size; ++b) {
        //            int data_idx = ((b * _out_channels + oc) * _in_width + x) * _in_height + y;
        //            grad_gamma += output_error[data_idx] * x_normalized[data_idx];
        //            grad_beta += output_error[data_idx];
        //        }

        //        atomicAdd(&gamma_gradient[oc], grad_gamma);
        //        atomicAdd(&beta_gradient[oc], grad_beta);
        //    }
        //    __syncthreads();
        //}
    }

    template<class T>
    Tensor<T>& BatchNorm<T>::backpropagation_4d(Tensor<T>& output_error, float learning_rate)
    {
        input_error.reset(new Tensor<T>({ input.getShape() }));
        grad_normalized.reset(new Tensor<T>({ input.getShape() }));
        const int _batch_size = input.getShape()[0];
        const int _out_channels = input.getShape()[1];
        const int _in_width = input.getShape()[2];
        const int _in_height = input.getShape()[3];

        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks(_batch_size * _out_channels,
            (_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        batchnorm_backward_4d_kernel << <numBlocks, threadsPerBlock >> > (input.getData(),
            output_error.getData(),
            x_normalized->getData(),
            input_mean.getData(),
            input_var.getData(),
            input_error->getData(),
            gamma.getData(),
            beta.getData(),
            grad_normalized->getData(),
            _batch_size,
            _out_channels,
            _in_width,
            _in_height,
            eps);
        cudaDeviceSynchronize();

        grad_normalized->print();
        return *input_error;
    }


 

    template class BatchNorm<float>;
    template class BatchNorm<int>;
    template class BatchNorm<double>;
}