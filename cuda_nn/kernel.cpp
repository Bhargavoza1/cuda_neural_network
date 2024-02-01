#include <iostream>

#include <stdexcept>
 
#include"Tensor.h"
#include "tensor_oprations.cu"

using namespace Hex;
using namespace std;

template <class T>
void printtensor(const Tensor<T>& tensor1) {
    tensor1.print();
}



int main() {
    try {
        // Create tensors with different shapes [2, 3, 4] and [2, 3, 5]
        std::unique_ptr<Tensor<double>> tensorA(new Tensor<double>({ 2, 3, 4 }));
        std::unique_ptr<Tensor<int>> tensorB(new Tensor<int>({ 2, 3, 4 }));  // Different shape

        // Initialize input data for 3D tensors
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 4; ++k) {
                    float val = static_cast<float>(i * 3 * 4 + j * 4 + k);
                    tensorA->set({ i, j, k }, val * 0.2);
                }
            }
        }

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 5; ++k) {  // Different size in the last dimension
                    float val = static_cast<float>(i * 3 * 5 + j * 5 + k);
                    tensorB->set({ i, j, k }, 2 * val);
                }
            }
        }

        // Perform element-wise addition for 3D tensors
        auto tensorC = Hex::addTensor(*tensorA, *tensorB);

        std::cout << "Tensor C (A + B):" << std::endl;
        printtensor(*tensorC);
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        // Handle the error as needed
    }

    return 0;
}
