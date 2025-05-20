#include <thrust/device_vector.h>
#include <thrust/system/cuda/vector.h> // Ensure CUDA-specific vector is included
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <vector>

using namespace std;

// Functor to multiply only the first element of the tuple by 2
struct double_first_component {
    __host__ __device__
    thrust::tuple<int, int> operator()(thrust::tuple<int, int> t) {
        int a = thrust::get<0>(t);  // first vector
        int b = thrust::get<1>(t);  // second vector (unchanged)
        return thrust::make_tuple(a * 2, b);
    }
};

int main() {
    const int N = 5;

    // Initialize host vectors
    std::vector<int> h_A = {1, 2, 3, 4, 5};
    std::vector<int> h_B = {10, 20, 30, 40, 50};

    // Copy host vectors to device vectors
    thrust::device_vector<int> vecA(h_A.begin(), h_A.end());
    thrust::device_vector<int> vecB(h_B.begin(), h_B.end());

    // Apply transformation on the zipped vector
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(vecA.begin(), vecB.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(vecA.end(), vecB.end())),
        thrust::make_zip_iterator(thrust::make_tuple(vecA.begin(), vecB.begin())),
        double_first_component()
    );

    // Copy back to host to print
    std::vector<int> h_A_result(N), h_B_result(N);
    thrust::copy(vecA.begin(), vecA.end(), h_A_result.begin());
    thrust::copy(vecB.begin(), vecB.end(), h_B_result.begin());

    std::cout << "After transformation:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "A[" << i << "] = " << h_A_result[i] << ", B[" << i << "] = " << h_B_result[i] << "\n";
    }

    return 0;
}
