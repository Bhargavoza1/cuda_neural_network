#pragma once
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

namespace Hex {
    template <typename T>
    class Tensor {
    private:
        static_assert(std::is_same<T, float>::value || std::is_same<T, int>::value || std::is_same<T, double>::value,
            "Tensor class supports only float, int, or double types.");

        T* data;
        std::vector<int> shape;

    public:
        Tensor(const std::vector<int>& shape);
        ~Tensor();
        void set(const std::vector<int>& indices, T value);
        T get(const std::vector<int>& indices) const;
        void print() const;
        void setData(T* newData);
        std::vector<int> getShape() const;
        const T* getData() const;
        T* getData();

    private:
        int calculateIndex(const std::vector<int>& indices) const;
        void printHelper(const T* data, const std::vector<int>& shape, int dimension, std::vector<int> indices) const;
        std::vector<int> calculateIndices(int index) const;
    };
}
