#include <iostream>

 
#include <cuda_runtime.h>
#include"utils/Tensor.h"
#include "utils/tensor_oprations.h"
 
#include "models/MLP.h"
#include "costs/MSE.h"

using namespace Hex;
using namespace std;

template <class T>
void printtensor(const Tensor<T>& tensor1) {
    tensor1.print();
}

 
template<typename T>
void predictAndPrintResults( MLP<T>& model, const Tensor<T>& input_data, const Tensor<T>& target_data) {
    std::vector<int> input_shape = input_data.getShape();
    std::vector<int> target_shape = target_data.getShape();

    // Assuming num_samples is the first dimension of the input_data and target_data tensors
    int num_samples = input_shape[0];
    std::unique_ptr<Tensor<T>> sliced_tensor2  ;
    std::unique_ptr<Tensor<T>> transpose_tensor2  ;
    Tensor<T> inpurt_data ;
    Tensor<T>* predicted_output;
    for (int sample_index = 0; sample_index < num_samples; ++sample_index) {
         
        sliced_tensor2 = Hex::sliceFirstIndex(sample_index, input_data) ;
        transpose_tensor2 = Hex::transpose(*sliced_tensor2);
        inpurt_data = *transpose_tensor2;  
        predicted_output =  &model.forward(inpurt_data);

        // Printing the predicted output
        std::cout << "Predicted output:" << std::endl;
        predicted_output->print();
 
        // Determine the actual output from the target_data
        int actual_output = (target_data.get({ sample_index, 0, 0 }) == 1) ? 0 : 1;
 
        // Print additional information
        if (predicted_output->get({ 0, 0 }) >= 0.5) {
            std::cout << "1st input for XOR: "<< inpurt_data.get({0,0}) <<" 2nd input for XOR: " << inpurt_data.get({ 0,1 }) << 
              " Neural Network output is 0, actual output is: " << actual_output << std::endl;
        }
        else {
            std::cout << "1st input for XOR: " << inpurt_data.get({ 0,0 }) << " 2nd input for XOR: " << inpurt_data.get({ 0,1 }) <<
                " Neural Network output is 1, actual output is: " << actual_output << std::endl;
        }

        std::cout << "end of cycle" << std::endl;
        std::cout << std::endl;
     
        ////// Print the sliced tensor
        //std::cout << "Sliced tensor at index " << sample_index << ":" << std::endl;
        //sliced_tensor->print();
        
    }
}

template<typename T>
void trainNeuralNetwork(MLP<T>& model, const Tensor<T>& input_data, const Tensor<T>& target_data, int num_epochs, T learning_rate) {
    // Get the number of samples
    std::vector<int> input_shape = input_data.getShape();
    std::vector<int> target_shape = target_data.getShape();

    int num_samples = input_shape[0];
    std::unique_ptr<Tensor<T>> input_slicing;
    std::unique_ptr<Tensor<T>> input_transpose;
    Tensor<T> sampled_input_data;
    
    std::unique_ptr<Tensor<T>> Target_slicing;
    std::unique_ptr<Tensor<T>> Target_transpose;
    Tensor<T> sampled_target_data;
    
    Tensor<T>* predicted_output; 

    std::unique_ptr<Tensor<T>> up_error;
    Tensor<T> error;

    std::unique_ptr<Tensor<T>> up_output_error;
    Tensor<T> output_error;
  


    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
      
        T total_error = 0;
        for (int sample_index = 0; sample_index < num_samples; ++sample_index) {
            
            input_slicing = Hex::sliceFirstIndex(sample_index, input_data);
            input_transpose = Hex::transpose(*input_slicing);
            sampled_input_data = *input_transpose;
            
            Target_slicing = Hex::sliceFirstIndex(sample_index, target_data);
            Target_transpose = Hex::transpose(*Target_slicing);
            sampled_target_data = *Target_transpose;
          
            predicted_output = &model.forward(sampled_input_data);
            //predicted_output->print();
            up_error = Hex::mse(sampled_target_data, *predicted_output);
            error = *up_error;
           // std::cout << error.get({ 0 }) << endl;
            total_error += error.get({0});  

            up_output_error = Hex::mse_derivative(sampled_target_data, *predicted_output);
            output_error = *up_output_error;
           // output_error.print();
            // Backward propagation
            model.backpropa(output_error, learning_rate);
        } 
        // Calculate the average error on all samples
        T average_error = (total_error / num_samples)  ;
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << "   Mean Squared Error: " << average_error << std::endl;
    } 
    std::cout << std::endl;
}

#include "layers/CNN2D.h"
int main() {
    
    // Define the parameters for your MLP
    int input_size = 2;        // Size of input layer
    int output_size = 2;       // Size of output layer
    int hiddenlayer = 1;       // Number of hidden layers
    int h_l_dimension = 3;     // Dimension of each hidden layer

    // Create an instance of the MLP class
    std::unique_ptr<Hex::MLP<float>>  mlp(new  Hex::MLP<float>(input_size, output_size, hiddenlayer, h_l_dimension));
    

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

   // trainNeuralNetwork(*mlp, *x_tensor, *y_tensor, 1000, 0.15f); 
  //  predictAndPrintResults(*mlp, *x_tensor, *y_tensor);
 
    std::vector<int> shape = {1 ,1,10,10 };
    Hex::Tensor<float> tensor(shape);

    // Assign sequential values starting from 1
    //float value = 0.0f;
    //for (int k = 0; k < shape[0]; ++k) {
    //    for (int f = 0; f < shape[1]; ++f) {
    //        for (int i = 0; i < shape[2]; ++i) {
    //            for (int j = 0; j < shape[3]; ++j) {
    //                tensor.set({k, f, i, j }, value);
    //                value += 1.0f;
    //            }
    //        }
    //    }
    //}
    initTensorOnGPU(tensor , 0.0f);
     //tensor.print();
        //std::cout << "input tensor" << endl;
        //tensor.print();
    CNN2D<float> convo(shape,{1,1},3 );
    Tensor<float>* predicted_output = &convo.forward(tensor);
     Tensor<float>* error_output = &convo.backpropagation(*predicted_output);

     std::cout << "predicted_output" << endl;
   
        predicted_output->print();
     std::cout << "after back propagation of predicted_output" << endl;
     std::cout   << endl;
     std::cout   << endl;
     std::cout   << endl;
        error_output->print();
    return 0;
 
   
     
}
