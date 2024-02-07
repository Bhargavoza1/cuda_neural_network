#include "MLP.h"
#include <iostream>
 
namespace Hex
{
	template <class T>
	MLP<T>::MLP(int input_size, int output_size, int hiddenlayer  , int h_l_dimension  ):_hiddenlayer(hiddenlayer) {
		linear1 = std::make_unique<linear<T>>(input_size, h_l_dimension);
		relu1 = std::make_unique<ReLU<T>>();
		linear2 = std::make_unique<linear<T>>(h_l_dimension, h_l_dimension);
		relu2 = std::make_unique<ReLU<T>>();
		linear3 = std::make_unique<linear<T>>(h_l_dimension, output_size, false);
		sigmoid1 = std::make_unique<Sigmoid<T>>();
	 
	}
	template<class T>
	MLP<T>::~MLP()
	{    
 
	}
	template<class T>
	Tensor<T>& MLP<T>::forward(Tensor<T>& input_tensor)
	{

		X = linear1->forward(input_tensor);
		//X.print();
		X = relu1->forward(X);

		//////// hidden layer
		for (int i = 0; i < _hiddenlayer; ++i) {
			X = linear2->forward(X);
			//X.print();
			X = relu2->forward(X);
			//X.print();
		}
		
		X = linear3->forward(X);
		//X.print();
		X = sigmoid1->forward(X);
		//X.print();
		return X;
	}

	template<class T>
	Tensor<T>& MLP<T>::backpropagation(Tensor<T>& output_error, float learning_rate  ) { return output_error; }

	template<class T>
	void MLP<T>::backpropa(Tensor<T>& output_error, float learning_rate  )
	{
		// Calculate gradients for the output layer
		X= sigmoid1->backpropagation(output_error, learning_rate);
		//X.print();
		X = linear3->backpropagation(X, learning_rate);
		//X.print();
	 
		for (int i = 0; i < _hiddenlayer; ++i) {
			X = relu2->backpropagation(X, learning_rate);
			//X.print();
			X = linear2->backpropagation(X, learning_rate);
			//X.print();
		}
		//X.print();
		// Backpropagate through the first hidden layer
		X = relu1->backpropagation(X, learning_rate);
		//X.print();
		X = linear1->backpropagation(X, learning_rate);
		//X.print();
		 
	}


	// Explicit instantiation of the template class for supported types
	template class MLP<float>;
	template class MLP<int>;
	template class MLP<double>;
}