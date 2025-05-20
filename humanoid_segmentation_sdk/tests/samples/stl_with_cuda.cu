#include <iostream>
#include <vector>
#include <cuda_runtime.h>
/**
 * @file
 * @brief This file contains CUDA C++ code designed for parallel computing on NVIDIA GPUs.
 *
 * It includes kernel functions and host code to perform high-performance computations
 * by leveraging the massively parallel architecture of CUDA-enabled devices.
 * The file is structured to optimize execution efficiency and memory usage.
 */

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA kernel that adds elements from A and B into C
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

class CudaMemory {
public:
    CudaMemory(size_t size) {
        CUDA_CHECK(cudaMalloc(&ptr, size));
    }
    ~CudaMemory() {
        cudaFree(ptr);
    }
    void* get() const { return ptr; }
private:
    void* ptr;
};

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    // Use STL vectors on host
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N);  // Result vector

    // Allocate memory on the device
    CudaMemory d_A(size);
    CudaMemory d_B(size);
    CudaMemory d_C(size);

    // Copy data from host STL vectors to device
    CUDA_CHECK(cudaMemcpy(d_A.get(), h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B.get(), h_B.data(), size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(static_cast<const float*>(d_A.get()), static_cast<const float*>(d_B.get()), static_cast<float*>(d_C.get()), N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to STL vector
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C.get(), size, cudaMemcpyDeviceToHost));

    // Check result
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != 3.0f) {
            std::cerr << "Mismatch at index " << i << ": " << h_C[i] << std::endl;
            break;
        }
    }

    std::cerr << "vector addition complete" << std::endl;
    return 0;
}
