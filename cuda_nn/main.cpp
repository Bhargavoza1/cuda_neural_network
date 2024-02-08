#include <iostream>

#include <stdexcept>
#include <cuda_runtime.h>
#include"Tensor.h"
#include "tensor_oprations.h"
#include "linear.h"
#include "ReLU.h"
#include "Sigmoid.h"
#include "MLP.h"
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
    std::unique_ptr<Tensor<float>> sliced_tensor  ;
    std::unique_ptr<Tensor<float>> transpose_tensor  ;
    Tensor<float> inpurt_data ;
    for (int sample_index = 0; sample_index < num_samples; ++sample_index) {
         
        sliced_tensor = Hex::sliceFirstIndex(sample_index, input_data) ;
        transpose_tensor = Hex::transpose(*sliced_tensor);
        inpurt_data = *transpose_tensor;
        //inpurt_data.print();
        auto a =  model.forward(inpurt_data); 
        model.backpropa(a);
        std::cout << "end of cycle";
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        ////// Print the sliced tensor
        //std::cout << "Sliced tensor at index " << sample_index << ":" << std::endl;
        //sliced_tensor->print();
 
    }
}
int main() {
    
    // Define the parameters for your MLP
    int input_size = 2;        // Size of input layer
    int output_size = 2;       // Size of output layer
    int hiddenlayer = 1;       // Number of hidden layers
    int h_l_dimension = 3;     // Dimension of each hidden layer

    // Create an instance of the MLP class
    Hex::MLP<float> mlp(input_size, output_size, hiddenlayer, h_l_dimension);

    // Define your input data
    std::vector<std::vector<std::vector<int>>> x_train = {
        {{0, 0}},
        {{0, 1}},
        {{1, 0}},
        {{1, 1}}
    };

    std::vector<std::vector<std::vector<int>>> y_train = {
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

    
    predictAndPrintResults(mlp, *x_tensor, *y_tensor);
    //// Print tensors
    //std::cout << "x_train:" << std::endl;
    //x_tensor->print();

    //std::cout << "y_train:" << std::endl;
    //y_tensor->print();

    //std::unique_ptr<Tensor<float>> sliced_tensor = Hex::sliceFirstIndex(1, *x_tensor);
    //std::unique_ptr<Tensor<float>> transpose_tensor = Hex::transpose(*sliced_tensor);
    //Tensor<float> inpurt_data = *transpose_tensor;
    //inpurt_data.print();

    //std::unique_ptr<Tensor<float>> x_2(new Tensor<float>({2,1}));
    //initTensorOnGPU(*x_2 ,0.0f);
    //x_2->print();
    // mlp.forward(inpurt_data);


    // Get the sliced tensor at index 0

 

    

    return 0;
 
   
     
}
