#include <iostream>
#include "example/XOR.h"
#include "models/Image_CF.h"
#include "utils/Tensor.h"
#include "utils/tensor_oprations.h"
#include "layers/BatchNorm.h"
#include "layers/linear.h"
#include "models/MLP.h"
#include "costs/MSE.h"
#include <opencv2/opencv.hpp>

#include "layers/CNN2D.h"

#include <numeric>
#include <algorithm> // for std::shuffle
#include <random>    // for std::default_random_engine
#include <chrono>    // for std::chrono::system_clock

using namespace std;
using namespace Hex;

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

        

        std::cout << "Epoch " << (i+1) << "/" << epoch << "   Mean Squared Error: " << error->get({ 0 }) << std::endl;
      
    }
    a.print();
}

void trainTestSplit(const std::vector<cv::String>& allFilePaths, float trainRatio,
    std::vector<cv::String>& trainFilePaths, std::vector<cv::String>& testFilePaths) {
    // Shuffle the file paths
    std::vector<cv::String> shuffledFilePaths = allFilePaths;
    std::shuffle(shuffledFilePaths.begin(), shuffledFilePaths.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

    // Calculate split indices
    size_t splitIndex = static_cast<size_t>(trainRatio * shuffledFilePaths.size());

    // Assign training and testing file paths
    trainFilePaths.assign(shuffledFilePaths.begin(), shuffledFilePaths.begin() + splitIndex);
    testFilePaths.assign(shuffledFilePaths.begin() + splitIndex, shuffledFilePaths.end());
}

void imagepreprocess(int width, int height, std::string normalPath, std::vector<cv::String> filepaths, std::vector<std::vector<int>>& lable, std::vector<cv::Mat>& images) {

    for (const auto& filePath : filepaths) {
        cv::Mat image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << filePath << std::endl;
            return;
        }

        // Resize the image to the specified width and height
        cv::resize(image, image, cv::Size(width, height));

        // Convert single channel grayscale to 3-channel grayscale
        cv::Mat colorImage;
        cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);

        // Normalize pixel values to the range [0, 1]
        cv::Mat normalizedImage;
        colorImage.convertTo(normalizedImage, CV_32FC3, 1.0 / 255.0);

        // Determine label based on folder
        std::string formattedPath = filePath;
        std::replace(formattedPath.begin(), formattedPath.end(), '\\', '/');
        int label = (formattedPath.find(normalPath) != std::string::npos) ? 0 : 1;
        // One-hot encode the label
        std::vector<int> labelOneHot(2, 0); // Initialize one-hot encoded label with zeros
        labelOneHot[label] = 1; // Set the appropriate index to 1 based on the label
        lable.push_back(labelOneHot); // Push the one-hot encoded label into the vector

        images.push_back(normalizedImage);

    }

}

void trainNeuralNetwork2(Image_CF<float>& model , std::vector<cv::Mat>& input , std::vector<std::vector<int>>& target,
    int batch, int channels, int num_epochs, float learning_rate)
{
    // Create a Tensor object to store the images 
    int numImages = input.size();
    int height = input[0].rows;
    int width = input[0].cols;

    int numBatches = (numImages + batch - 1) / batch;

    std::vector<int> shape = { batch , channels , height, width };
    Hex::Tensor<float> imageTensor(shape);
    Hex::Tensor<float> labelTensor({ batch,2 });

    // Copy image data from CPU to GPU
    size_t size = batch * height * width * channels * sizeof(float);
    float* gpuData;
    cudaError_t cudaStatus = cudaMalloc((void**)&gpuData, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return ;
    }

    size_t labelSize = batch * 2 * sizeof(float);
    float* gpuLabelData;
    cudaError_t cudaStatusLabel = cudaMalloc((void**)&gpuLabelData, labelSize);
    if (cudaStatusLabel != cudaSuccess) {
        std::cerr << "cudaMalloc for label tensor failed: " << cudaGetErrorString(cudaStatusLabel) << std::endl;
        cudaFree(gpuLabelData); // Free the previously allocated memory
        return  ;
    }
    std::shared_ptr<Tensor<float>> error;
    std::shared_ptr<Tensor<float>> output_error;
    Tensor<float> a;
    
    for (int e = 0; e < num_epochs; e++) {
        float total_error = 0;
        for (int i = 0; i < numBatches; ++i) {
            for (int j = 0; j < batch; ++j) {
                int index = i * batch + j;
                if (index < numImages) {
                    cudaStatus = cudaMemcpy(gpuData + j * height * width * channels, input[index].data, size / batch, cudaMemcpyHostToDevice);
                    if (cudaStatus != cudaSuccess) {
                        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                        cudaFree(gpuData);
                        return  ;
                    }

                    for (int k = 0; k < target[index].size(); ++k) {
                        float labelValue = static_cast<float>(target[index][k]);
                        cudaStatus = cudaMemcpy(gpuLabelData + j * 2 + k, &labelValue, sizeof(float), cudaMemcpyHostToDevice);
                        if (cudaStatus != cudaSuccess) {
                            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                            cudaFree(gpuData);
                            cudaFree(gpuLabelData);
                            return ;
                        }
                    }
                }
            }


            imageTensor.setData(gpuData);

            labelTensor.setData(gpuLabelData);



            a = model.forward(imageTensor);
            error = Hex::mse(labelTensor, a);

            total_error += error->get({ 0 });

            output_error = Hex::mse_derivative(labelTensor, a);
            model.backpropa(*output_error, learning_rate);

        }
        float average_error = (total_error / numBatches);
        std::cout << "Epoch " << (e + 1) << "/" << num_epochs << "   Mean Squared Error: " << average_error << std::endl;
       // a.print();
        //labelTensor.print();

       // std::cout << std::endl;
    }

    cudaFree(gpuData);
    cudaFree(gpuLabelData);

}

void testNeuralNetwork2(Image_CF<float>& model, std::vector<cv::Mat>& input, std::vector<std::vector<int>>& target , std::vector<cv::String> filepaths)
{
    // Create a Tensor object to store the images 
    int numImages = input.size();
    int height = input[0].rows;
    int width = input[0].cols;
    int channels = 3;
    int batch = 1;

    std::vector<int> shape = { batch , channels , height, width };
    Hex::Tensor<float> imageTensor(shape);
    Hex::Tensor<float> labelTensor({ batch,2 });

    // Copy image data from CPU to GPU
    size_t size =   height * width * channels * sizeof(float);
    float* gpuData;
    cudaError_t cudaStatus = cudaMalloc((void**)&gpuData, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    size_t labelSize = batch * 2 * sizeof(float);
    float* gpuLabelData;
    cudaError_t cudaStatusLabel = cudaMalloc((void**)&gpuLabelData, labelSize);
    if (cudaStatusLabel != cudaSuccess) {
        std::cerr << "cudaMalloc for label tensor failed: " << cudaGetErrorString(cudaStatusLabel) << std::endl;
        cudaFree(gpuLabelData); // Free the previously allocated memory
        return;
    } 

    Tensor<float> a;

    for (int image_x = 0; image_x < numImages; ++image_x) {
        cudaStatus = cudaMemcpy(gpuData  , input[image_x].data, size   , cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(gpuData);
            return;
        }

        for (int k = 0; k < target[image_x].size(); ++k) {
            float labelValue = static_cast<float>(target[image_x][k]);
            cudaStatus = cudaMemcpy(gpuLabelData + k, &labelValue, sizeof(float), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                cudaFree(gpuData);
                cudaFree(gpuLabelData);
                return;
            }
        }

        imageTensor.setData(gpuData);

        labelTensor.setData(gpuLabelData);

        //imageTensor.printshape();
         //labelTensor.print();

         
            std::cout << filepaths[image_x] << endl;
         
         a = model.forward(imageTensor ,  false);
         a.print();
         labelTensor.print();
    }


 
}

int main() {

    //Hex::xor_example();
 
    std::string normalPath = "../../kidney-ct-scan-image/Normal/";
    std::string tumorPath = "../../kidney-ct-scan-image/Tumor/";

    // Load file paths from the Normal directory
    std::vector<cv::String> normalFilePaths;
    cv::glob(normalPath + "*.jpg", normalFilePaths);

    // Load file paths from the Tumor directory
    std::vector<cv::String> tumorFilePaths;
    cv::glob(tumorPath + "*.jpg", tumorFilePaths);

    // Combine all file paths into one vector
    std::vector<cv::String> allFilePaths;
    allFilePaths.insert(allFilePaths.end(), normalFilePaths.begin(), normalFilePaths.end());
    allFilePaths.insert(allFilePaths.end(), tumorFilePaths.begin(), tumorFilePaths.end());



    float trainRatio = 0.981;
    std::vector<cv::String> trainFilePaths;
    std::vector<cv::String> testFilePaths;
   
    // Perform train-test split
    trainTestSplit(allFilePaths, trainRatio, trainFilePaths, testFilePaths);

 

    std::vector<cv::Mat> train_Images;
    std::vector<std::vector<int>>  train_LabelsOneHot;

    std::vector<cv::Mat> test_Images;
    std::vector<std::vector<int>>  test_LabelsOneHot;

    int resizeWidth = 225;
    int resizeHeight = 225;
    int channels = 3;

    /////imagepreprocess for train data
    imagepreprocess(resizeWidth, resizeHeight, normalPath, trainFilePaths, train_LabelsOneHot, train_Images);

    /////imagepreprocess for test data
    imagepreprocess(resizeWidth, resizeHeight, normalPath, testFilePaths, test_LabelsOneHot, test_Images);

    //cout << trainFilePaths.size() << endl;
   // cout << test_LabelsOneHot.size() << endl;

    int batchSize = 8;
    int epoch = 20;

    std::unique_ptr<Hex::Image_CF<float>>  Image_CF(new  Hex::Image_CF<float>(batchSize, channels, 2));

     trainNeuralNetwork2(*Image_CF, train_Images, train_LabelsOneHot, batchSize , channels , epoch , 0.00001f);


    testNeuralNetwork2(*Image_CF, test_Images, test_LabelsOneHot , testFilePaths);




    return 0;

}
