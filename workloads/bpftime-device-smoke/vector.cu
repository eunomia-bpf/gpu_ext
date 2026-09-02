#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) do { const cudaError_t error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); return 2; } } while (0)

__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    constexpr int n = 4096;
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(3 * i);
    }
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    unsigned checked = 0;
    for (int launch = 0; launch < 8; ++launch) {
        vectorAdd<<<32, 128>>>(da, db, dc, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            if (c[i] != static_cast<float>(4 * i)) {
                std::fprintf(stderr, "mismatch launch=%d index=%d got=%g expected=%d\n", launch, i, c[i], 4 * i);
                return 3;
            }
            ++checked;
        }
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    std::printf("{\"event\":\"correctness\",\"launches\":8,\"checked_values\":%u,\"mismatches\":0}\n", checked);
    return 0;
}
