
#include"MaxPool2d.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
namespace Hex {
	template<class T>
	MaxPool2d<T>::MaxPool2d(int kernel_size, int stride, int padding):
		_kernel_size(kernel_size) , _padding(padding) , _stride(stride) 
	{
	}

	template<class T>
	MaxPool2d<T>::~MaxPool2d()
	{
	}

	template <typename T>
	__global__ void maxpool2d_forward_kernel(const T* input, T* output,
		int batch_size, int channel, int input_width, int input_height,
		int output_width, int output_height,
		int kernel_size, int stride , int padding) {

	}

	template<class T>
	Tensor<T>& MaxPool2d<T>::forward(Tensor<T>& input_tensor)
	{
		input = input_tensor;


		int  _batch_size = input.getShape()[0];
		int  _channel_size = input.getShape()[0];
		int  _in_width = input.getShape()[2];
		int  _in_height = input.getShape()[3];

		int _out_width = ((_in_width - _kernel_size + 2 * _padding) / _stride) + 1;
		int _out_height = ((_in_height - _kernel_size + 2 * _padding) / _stride) + 1;

		//std::cout << _in_width << _in_height << _out_width << _out_height;
		// in pooling we are only changging _out_width and _out_height other value will stay same 
		output.reset(new Tensor<T>({ _batch_size , _channel_size ,_out_width , _out_height }));

		dim3 threadsPerBlock(8, 8, 8);
		dim3 numBlocks(_batch_size * _channel_size,
			(_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y);  

		maxpool2d_forward_kernel << <numBlocks, threadsPerBlock >> > (input.getData(), output->getData(),
			_batch_size , _channel_size , _in_width, _in_height,
			_out_width, _in_height,
			_kernel_size, _stride, _padding);

		return input_tensor;
	}
	template<class T>
	Tensor<T>& MaxPool2d<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
	{
		return output_error;
	}

	template class MaxPool2d<float>;
	template class MaxPool2d<int>;
	template class MaxPool2d<double>;
}