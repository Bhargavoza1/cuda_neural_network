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


int main() {

    //Hex::xor_example();

   //xor_withcnn();


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
     // Specify the directory paths
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

    //std::cout << allFilePaths.size() << std::endl;
    //std::cout << trainFilePaths.size() << std::endl;
    //std::cout << testFilePaths.size() << std::endl;


    std::vector<cv::Mat> train_Images;
    std::vector<std::vector<int>>  train_LabelsOneHot;

    std::vector<cv::Mat> test_Images;
    std::vector<std::vector<int>>  test_LabelsOneHot;

    int resizeWidth = 225;
    int resizeHeight = 225;
    int channels = 3;

    imagepreprocess(resizeWidth, resizeHeight, normalPath, trainFilePaths, train_LabelsOneHot, train_Images);

    imagepreprocess(resizeWidth, resizeHeight, normalPath, testFilePaths, test_LabelsOneHot, test_Images);

    // Create a Tensor object to store the images 
    int numImages = train_Images.size();
    int height = train_Images[0].rows;
    int width = train_Images[0].cols;


    int batchSize = 8;
    int numBatches = (numImages + batchSize - 1) / batchSize;

    std::unique_ptr<Hex::Image_CF<float>>  Image_CF(new  Hex::Image_CF<float>(batchSize, channels, 2));

    std::vector<int> shape = { batchSize , channels , height, width };
    Hex::Tensor<float> imageTensor(shape);
    Hex::Tensor<float> labelTensor({ batchSize,2 });

    // Copy image data from CPU to GPU
    size_t size = batchSize * height * width * channels * sizeof(float);
    float* gpuData;
    cudaError_t cudaStatus = cudaMalloc((void**)&gpuData, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    size_t labelSize = batchSize * 2 * sizeof(float);
    float* gpuLabelData;
    cudaError_t cudaStatusLabel = cudaMalloc((void**)&gpuLabelData, labelSize);
    if (cudaStatusLabel != cudaSuccess) {
        std::cerr << "cudaMalloc for label tensor failed: " << cudaGetErrorString(cudaStatusLabel) << std::endl;
        cudaFree(gpuLabelData); // Free the previously allocated memory
        return 1;
    }
    std::shared_ptr<Tensor<float>> error;
    std::shared_ptr<Tensor<float>> output_error;
    Tensor<float> a;
    int epoch = 20;
    for(int e = 0 ; e < epoch ; e++){
        float total_error = 0;
        for (int i = 0; i < numBatches; ++i) {
            for (int j = 0; j < batchSize; ++j) {
                int index = i * batchSize + j;
                if (index < numImages) {
                    cudaStatus = cudaMemcpy(gpuData + j * height * width * channels, train_Images[index].data, size / batchSize, cudaMemcpyHostToDevice);
                    if (cudaStatus != cudaSuccess) {
                        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                        cudaFree(gpuData);
                        return 1;
                    }

                    for (int k = 0; k < train_LabelsOneHot[index].size(); ++k) {
                        float labelValue = static_cast<float>(train_LabelsOneHot[index][k]);
                        cudaStatus = cudaMemcpy(gpuLabelData + j * 2 + k, &labelValue, sizeof(float), cudaMemcpyHostToDevice);
                        if (cudaStatus != cudaSuccess) {
                            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                            cudaFree(gpuData);
                            cudaFree(gpuLabelData);
                            return 1;
                        }
                    }
                }
            }

            // Store image data from GPU memory into the Tensor
            imageTensor.setData(gpuData);

            // Print the tensor (This will print a lot of data)
           // imageTensor.print();

            // Store label data from GPU memory into the label Tensor
            labelTensor.setData(gpuLabelData);

            // Print the label tensor (This will print a lot of data)
          //  labelTensor.print();

            a = Image_CF->forward(imageTensor);
         
            
           // a.print();
           // labelTensor.print();
                  error = Hex::mse(labelTensor, a);

                  total_error += error->get({ 0 });
               ////   a.print();
               output_error = Hex::mse_derivative(labelTensor, a);
               // // output_error->print();
                Image_CF->backpropa(*output_error, 0.00001f);



              

            

        }
        float average_error = (total_error / numBatches);
        std::cout << "Epoch " << (e + 1) << "/" << epoch << "   Mean Squared Error: " << average_error << std::endl;
         a.print();
         labelTensor.print();

        std::cout << std::endl;
    }
    // Print the tensor (This will print a lot of data)
    cudaFree(gpuData);
    cudaFree(gpuLabelData);




    return 0;

}
