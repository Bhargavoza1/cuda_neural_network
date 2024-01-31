// tensor.h

#pragma once
#include <cstdio> 
#include <vector>
#include <iostream>
#include <cuda_runtime.h> 

namespace Hex{



    template <typename T>
    class Tensor {
    private:
        T* data;
        std::vector<int> shape;

    public:
        // Constructor
        Tensor(const std::vector<int>& shape) : shape(shape) {
            int size = 1;
            for (int dim : shape) {
                size *= dim;
            }
            cudaMalloc((void**)&data, size * sizeof(T));
        }

        // Destructor
        ~Tensor() {
            cudaFree(data);
        }

        // Copy constructor
        Tensor(const Tensor& other) : shape(other.shape) {
            int size = 1;
            for (int dim : shape) {
                size *= dim;
            }
            cudaMalloc((void**)&data, size * sizeof(T));
            cudaMemcpy(data, other.data, size * sizeof(T), cudaMemcpyDeviceToDevice);
        }

        // Set element at index
        void set(const std::vector<int>& indices, T value) {
            int index = calculateIndex(indices);
            cudaMemcpy(data + index, &value, sizeof(T), cudaMemcpyHostToDevice);
        }

        // Get element at index
        T get(const std::vector<int>& indices) const {
            int index = calculateIndex(indices);
            T value;
            cudaMemcpy(&value, data + index, sizeof(T), cudaMemcpyDeviceToHost);
            return value;
        }

        // Print the tensor
        void print() const {
            std::cout << "Tensor (Shape: ";
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cout << shape[i];
                if (i < shape.size() - 1) {
                    std::cout << "x";
                }
            }
            std::cout << ", Type: " << typeid(T).name() << "):" << std::endl;

            printHelper(data, shape, 0, {});
        }

        void setData(T* newData) {
            data = newData;
        }
        // Getter for shape
        std::vector<int> getShape() const {
            return shape;
        }

        const T* getData() const {
            return data;
        }

        T* getData()   {
            return data;
        }
    private:
        // Helper function to calculate the flat index from indices
        int calculateIndex(const std::vector<int>& indices) const {
            int index = 0;
            int stride = 1;
            for (int i = shape.size() - 1; i >= 0; --i) {
                index += indices[i] * stride;
                stride *= shape[i];
            }
            return index;
        }

        void printHelper(const T* data, const std::vector<int>& shape, int dimension, std::vector<int> indices) const {
            int currentDimensionSize = shape[dimension];

            std::cout << "[";

            for (int i = 0; i < currentDimensionSize; ++i) {
                indices.push_back(i);

                if (dimension < shape.size() - 1) {
                    // If not the last dimension, recursively print the next dimension
                    printHelper(data, shape, dimension + 1, indices);
                }
                else {
                    // If the last dimension, print the actual element
                    std::cout << get(indices);
                }

                indices.pop_back();

                if (i < currentDimensionSize - 1) {
                    std::cout << ", ";
                }
            }

            std::cout << "]";

            if (dimension < shape.size() - 1) {
                // If not the last dimension, add a new line after completing the inner block
                std::cout << std::endl;
            }
        }

        // Helper function to calculate indices from a flat index
        std::vector<int> calculateIndices(int index) const {
            std::vector<int> indices(shape.size(), 0);
            for (int i = shape.size() - 1; i >= 0; --i) {
                indices[i] = index % shape[i];
                index /= shape[i];
            }
            return indices;
        }
    };
}
