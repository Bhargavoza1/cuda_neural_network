#include <iostream>
#include "example/XOR.h"
#include "models/Image_CF.h"
#include "utils/Tensor.h"
#include "utils/tensor_oprations.h"
#include "layers/BatchNorm.h"
#include "layers/linear.h"
#include "models/MLP.h"
#include "costs/MSE.h"
using namespace std;
using namespace Hex;
#include "layers/CNN2D.h"


void xor_withcnn() {
    int epoch = 50;
    std::vector<std::vector<std::vector<float>>> y_train = {
        {{1, 0}},   // Class 0
        {{0, 1}},   // Class 1
        {{0, 1}},   // Class 1
         {{1, 0}}  // Class 0
    };



    // Create a Tensor for y_train
    std::vector<int> y_shape = { 4, 1, 2 }; // Shape: (4, 1, 2)
    std::unique_ptr<Tensor<float>> y_tensor(new Tensor<float>(y_shape));

    // Set data for x_tensor
    for (int i = 0; i < 4; ++i) {


        y_tensor->set({ i, 0, 0 }, y_train[i][0][0]);
        y_tensor->set({ i, 0, 1 }, y_train[i][0][1]);
    }


    int batchsize = 4;
    int input_channels = 3;
    int output_class = 2;
    std::unique_ptr<Hex::Image_CF<float>>  Image_CF(new  Hex::Image_CF<float>(batchsize, input_channels, output_class));

    std::vector<int> x_shape = { batchsize,input_channels,512 ,512 }; // Shape: (4, 1, 2)


    std::unique_ptr<Tensor<float>> x_tensor(new Tensor<float>(x_shape));
    initTensorOnGPU(*x_tensor, 0.0f);
    //  x_tensor->print();
    y_tensor->reshape({ 4,2 });
    Tensor<float> a;
    std::shared_ptr<Tensor<float>> error;
    std::shared_ptr<Tensor<float>> output_error;
    for (int i = 0; i < epoch; i++) {
        a = Image_CF->forward(*x_tensor);
        error = Hex::mse(*y_tensor, a);


       // a.print();
        output_error = Hex::mse_derivative(*y_tensor, a);
        // output_error->print();
        Image_CF->backpropa(*output_error, 0.0001f);

        std::cout << " epoch " << i+1 << " done" << std::endl;

        std::cout << "Epoch " << (i+1) << "/" << epoch << "   Mean Squared Error: " << error->get({ 0 }) << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }
    a.print();
}


int main() {

    //Hex::xor_example();

   xor_withcnn();


    //int batchsize = 64;
    //int input_channels = 3;
    //int output_class = 1; 

    //std::vector<int> x_shape = { batchsize,input_channels,512 ,512 }; // Shape: (4, 1, 2)

    //std::unique_ptr<Tensor<float>> x_tensor(new Tensor<float>(x_shape));
    //initTensorOnGPU(*x_tensor, 0.0f);


    //BatchNorm<float> bc1(input_channels, TensorShape::_4D);
    //auto a = bc1.forward(*x_tensor);
    // bc1.backpropagation(*x_tensor); 
   // a.print();
    return 0;

}
