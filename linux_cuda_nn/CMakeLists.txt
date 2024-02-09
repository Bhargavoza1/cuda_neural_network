cmake_minimum_required(VERSION 3.12)
project(cuda_nn)

# Find CUDA package
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
list(APPEND CUDA_NVCC_FLAGS -arch=sm_89)

# Include directories for CUDA and other headers
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/costs
    ${CMAKE_CURRENT_SOURCE_DIR}/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/models
)

# Add executable target
set(SOURCES
    main.cpp
    costs/MSE.cu
    layers/linear.cu
    layers/ReLU.cu
    layers/Sigmoid.cu
    models/MLP.cpp
    utils/Errorhelper.cpp
    utils/Tensor.cu
    utils/tensor_oprations.cu
)

 
# Set CUDA language for CUDA sources
cuda_add_executable(cuda_nn ${SOURCES})
# Link CUDA libraries
target_link_libraries(cuda_nn ${CUDA_LIBRARIES})
