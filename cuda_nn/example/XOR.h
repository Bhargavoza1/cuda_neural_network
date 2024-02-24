#pragma once
#ifndef XOR_H
#define XOR_H

 
#include"../utils/Tensor.h"
#include "../utils/tensor_oprations.h"
#include "../models/MLP.h"
#include "../costs/MSE.h"

namespace Hex{

	template<typename T>
	void predictAndPrintResults(MLP<T>& model, const Tensor<T>& input_data, const Tensor<T>& target_data);

	template<typename T>
	void trainNeuralNetwork(MLP<T>& model, const Tensor<T>& input_data, const Tensor<T>& target_data, int num_epochs, T learning_rate);

	void xor_example();
}

#include "XOR.cu"

namespace Hex {

 


    void xor_example() {
        // Define the parameters for your MLP
        int input_size = 2;        // Size of input layer
        int output_size = 2;       // Size of output layer
        int batchsize = 4;
        int hiddenlayer = 3;       // Number of hidden layers
        int h_l_dimension = 15;     // Dimension of each hidden layer

        // Create an instance of the MLP class
        std::unique_ptr<Hex::MLP<float>>  mlp(new  Hex::MLP<float>(input_size, output_size, batchsize, hiddenlayer, h_l_dimension));


        // Define your input data
        std::vector<std::vector<std::vector<float>>> x_train = {
            {{0, 0}},
            {{0, 1}},
            {{1, 0}},
            {{1, 1}}
        };

        std::vector<std::vector<std::vector<float>>> y_train = {
            {{1, 0}},   // Class 0
            {{0, 1}},   // Class 1
            {{0, 1}},   // Class 1
             {{1, 0}}  // Class 0
        };

        // Create a Tensor for x_train
        std::vector<int> x_shape = { 4, 1, 2 }; // Shape: (4, 1, 2)
        std::unique_ptr<Tensor<float>> x_tensor(new Tensor<float>(x_shape));

        // Create a Tensor for y_train
        std::vector<int> y_shape = { 4, 1, 2 }; // Shape: (4, 1, 2)
        std::unique_ptr<Tensor<float>> y_tensor(new Tensor<float>(y_shape));
        // Set data for x_tensor
        for (int i = 0; i < 4; ++i) {
            x_tensor->set({ i, 0, 0 }, x_train[i][0][0]);
            x_tensor->set({ i, 0, 1 }, x_train[i][0][1]);

            y_tensor->set({ i, 0, 0 }, y_train[i][0][0]);
            y_tensor->set({ i, 0, 1 }, y_train[i][0][1]);
        }


        trainNeuralNetwork(*mlp, *x_tensor, *y_tensor, 100, 0.1f);
        predictAndPrintResults(*mlp, *x_tensor, *y_tensor);
    }
}
#endif 
