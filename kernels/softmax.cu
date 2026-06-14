// ===========================================================================
// softmax.cu  --  the attention softmax, and a look at how a *warp* cooperates.
// ===========================================================================
//
// Inside attention we do  softmax(QK^T / sqrt(d))  over each row.  A softmax is
// a *reduction* (find the row max, then the sum of exps) followed by a map.
// Reductions are where GPUs show their most distinctive trick: the 32 threads
// of a warp can exchange registers DIRECTLY with `__shfl_down_sync`, with no
// shared memory and no __syncthreads at all.
//
// This file assigns one warp (32 threads) to one row and reduces with shuffles.
// It is the kernel to dump to PTX/SASS to see the famous instruction:
//
//      shfl.sync.down.b32      <-- a register handed from lane i+offset to lane i
//
// Build & run:  nvcc -O3 -arch=sm_89 softmax.cu -o softmax && ./softmax
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define WARP 32

// Reduce a value across the 32 lanes of a warp to a single max (lane 0 holds it).
// Each step, lane i looks at lane i+offset's register and keeps the larger.
// After 5 steps (16,8,4,2,1) every lane has seen all 32 values.
__device__ __forceinline__ float warp_max(float v) {
    for (int off = WARP / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// One warp handles one row of length `n`.  Classic 3-pass numerically-stable
// softmax: (1) max for stability, (2) sum of exp(x-max), (3) normalise.
__global__ void softmax_rows(const float* X, float* Y, int rows, int n) {
    int row = blockIdx.x;                                    // one block = one warp = one row
    int lane = threadIdx.x;                                  // 0..31
    if (row >= rows) return;
    const float* x = X + (size_t)row * n;
    float* y = Y + (size_t)row * n;

    // pass 1: each lane scans a strided slice, then the warp reduces to the max
    float m = -1e30f;
    for (int i = lane; i < n; i += WARP) m = fmaxf(m, x[i]);
    m = warp_max(m);
    m = __shfl_sync(0xffffffff, m, 0);                       // broadcast lane 0 -> all

    // pass 2: sum of exp(x - max), again strided then warp-reduced
    float s = 0.f;
    for (int i = lane; i < n; i += WARP) s += expf(x[i] - m);
    s = warp_sum(s);
    s = __shfl_sync(0xffffffff, s, 0);

    // pass 3: write the normalised probabilities
    float inv = 1.f / s;
    for (int i = lane; i < n; i += WARP) y[i] = expf(x[i] - m) * inv;
}

static void cpu_softmax(const float* X, float* Y, int rows, int n) {
    for (int r = 0; r < rows; ++r) {
        const float* x = X + r * n; float* y = Y + r * n;
        float m = -1e30f; for (int i = 0; i < n; ++i) m = fmaxf(m, x[i]);
        float s = 0.f; for (int i = 0; i < n; ++i) s += expf(x[i] - m);
        for (int i = 0; i < n; ++i) y[i] = expf(x[i] - m) / s;
    }
}

int main() {
    int rows = 1024, n = 256;                                // e.g. 1024 attention rows
    size_t sz = (size_t)rows * n;
    float *hX = (float*)malloc(sz * 4), *hY = (float*)malloc(sz * 4);
    float *hRef = (float*)malloc(sz * 4);
    for (size_t i = 0; i < sz; ++i) hX[i] = (float)rand() / RAND_MAX * 6.f - 3.f;

    float *dX, *dY;
    cudaMalloc(&dX, sz * 4); cudaMalloc(&dY, sz * 4);
    cudaMemcpy(dX, hX, sz * 4, cudaMemcpyHostToDevice);

    softmax_rows<<<rows, WARP>>>(dX, dY, rows, n);           // one warp per row
    cudaDeviceSynchronize();
    cudaMemcpy(hY, dY, sz * 4, cudaMemcpyDeviceToHost);

    cpu_softmax(hX, hRef, rows, n);
    float err = 0.f; for (size_t i = 0; i < sz; ++i) err = fmaxf(err, fabsf(hY[i] - hRef[i]));
    // every row must sum to 1
    float rowsum = 0.f; for (int i = 0; i < n; ++i) rowsum += hY[i];
    printf("warp-softmax: max|err vs CPU|=%.2e   row0 sum=%.5f (should be 1)\n", err, rowsum);

    free(hX); free(hY); free(hRef); cudaFree(dX); cudaFree(dY);
    return 0;
}
