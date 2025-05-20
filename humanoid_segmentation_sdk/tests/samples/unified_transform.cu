#include <iostream>
#include <vector>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <functional>  // for std::plus

// ================================
// Templated Transform Function
// ================================

template <typename ExecPolicy, typename Vec>
void transform_add(const ExecPolicy& policy, const Vec& A, const Vec& B, Vec& C) {
    using T = typename Vec::value_type;
    thrust::transform(policy, A.begin(), A.end(), B.begin(), C.begin(), thrust::plus<T>());
}

// ================================
// Main
// ================================

int main() {
    const int N = 5;

    // -------- CPU Version using std::vector + thrust::host
    std::vector<int> h_A(N, 1);
    std::vector<int> h_B(N, 2);
    std::vector<int> h_C(N);

    transform_add(thrust::host, h_A, h_B, h_C);
    std::cout << "CPU Result:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_C[i] << " ";
    std::cout << "\n";

    // -------- GPU Version using thrust::device_vector
    thrust::device_vector<int> d_A(N, 3);
    thrust::device_vector<int> d_B(N, 4);
    thrust::device_vector<int> d_C(N);

    transform_add(thrust::device, d_A, d_B, d_C);

    // Copy to host for display
    std::vector<int> result(N);
    thrust::copy(d_C.begin(), d_C.end(), result.begin());

    std::cout << "GPU Result:\n";
    for (int i = 0; i < N; ++i)
        std::cout << result[i] << " ";
    std::cout << "\n";

    return 0;
}
