# Neural networks from scratch in CUDA C++

In this project, I tried to make performance similar to PyTorch. but there is still a long way to go.  
I was using rtx 4070ti gpu in my local computer.

# Algo that I have written in this project.

## Layers
- [x] BatchNorm
- [x] CNN2D
- [x] flatten_layer
- [x] linear
- [x] MaxPool2d
- [x] ReLU
- [x] Sigmoid

## costs
- [x] MSE(mean squared error)	

## Models 
- [x] MLP(multilayer perceptron) 
- [x] Image_CF(image classification).
Used **opencv** lib for image processing.
  
## Image classification Model architecture
<p align="center">
  <img src="./gitresource/image_classification.png" />
</p>


# How to run this?

## For windows
I have used Visual Studio 2019.  
MSVC = 14.29.30133 (v142)  
CUDA = v12.2  

After installing all those tools. It will run without any problem.  

## For Linux 
Copy **linux_cuda_nn** to your any dir.

``` 
cd linux_cuda_nn
mkdir build
cd build
cmake ..
make
./cuda_nn
```


