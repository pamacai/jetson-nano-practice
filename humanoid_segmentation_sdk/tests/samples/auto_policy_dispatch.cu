#include <iostream>
#include <vector>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <functional>

// =================================================================
// Policy Detection
// =================================================================

template<typename T>
struct exec_policy_selector;

template<typename T>
struct exec_policy_selector<std::vector<T>> {
    static auto policy() {
        return thrust::host; // Return thrust::host directly
    }
};

template<typename T>
struct exec_policy_selector<thrust::host_vector<T>> {
    static auto policy() {
        return thrust::host; // Return thrust::host directly
    }
};

template<typename T>
struct exec_policy_selector<thrust::device_vector<T>> {
    static auto policy() {
        return thrust::device; // Return thrust::device directly
    }
};

// =================================================================
// Unified Transform Function
// =================================================================

template<typename Vec>
void transform_add(const Vec& A, const Vec& B, Vec& C) {
    auto exec_policy = exec_policy_selector<Vec>::policy();
    using T = typename Vec::value_type;

    thrust::transform(exec_policy, A.begin(), A.end(), B.begin(), C.begin(), thrust::plus<T>());
}

// =================================================================
// Main
// =================================================================

int main() {
    const int N = 5;

    // -------- CPU (std::vector)
    std::vector<int> h_A(N, 1), h_B(N, 2), h_C(N);
    transform_add(h_A, h_B, h_C);
    std::cout << "CPU (std::vector) Result: ";
    for (int v : h_C) std::cout << v << " ";
    std::cout << "\n";

    // -------- Host Vector (thrust::host_vector)
    thrust::host_vector<int> th_A(N, 3), th_B(N, 4), th_C(N);
    transform_add(th_A, th_B, th_C);
    std::cout << "Host Vector Result: ";
    for (int v : th_C) std::cout << v << " ";
    std::cout << "\n";

    // -------- GPU (thrust::device_vector)
    thrust::device_vector<int> d_A(N, 5), d_B(N, 6), d_C(N);
    transform_add(d_A, d_B, d_C);
    std::vector<int> gpu_result(N);
    thrust::copy(d_C.begin(), d_C.end(), gpu_result.begin());
    std::cout << "GPU (device_vector) Result: ";
    for (int v : gpu_result) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
