// ===========================================================================
// matmul.cu  --  the matrix multiply, on the GPU, demystified.
// ===========================================================================
//
// Every `@` in model.py ultimately becomes a matrix multiply, and matrix
// multiply is where essentially all of a transformer's FLOPs go.  This file
// implements C = A * B (A is MxK, B is KxN) two ways and proves both correct
// against a plain CPU loop, then times them on your GPU.
//
//   1. sgemm_naive  -- one thread per output element, every thread reads its
//                      whole row of A and column of B straight from slow
//                      global memory.  Simple, but memory-bound.
//
//   2. sgemm_tiled  -- threads cooperate in 16x16 blocks, staging tiles of A
//                      and B into fast on-chip __shared__ memory and reusing
//                      each loaded value 16 times.  This is the single most
//                      important optimisation in GPU linear algebra.
//
// Build & run:   ./build.sh   (or: nvcc -O3 -arch=sm_89 matmul.cu -o matmul)
// See the actual GPU assembly:   ./dump.sh   then read web/asm/*.ptx / *.sass
//
// Read the comments top-to-bottom; the goal is that the leap from `A @ B` in
// Python to warps moving floats through shared memory feels continuous.
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 16                                   // 16x16 = 256 threads per block

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e){ \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

// ---------------------------------------------------------------------------
// Kernel 1: the obvious version.  Thread (row,col) computes one C[row,col].
// It walks the full K dimension, reading A[row,*] and B[*,col] from global
// memory.  Adjacent threads re-read the same data, so DRAM bandwidth is the
// bottleneck -- this is the baseline the tiled version will beat.
// ---------------------------------------------------------------------------
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];          // two global loads / FMA
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: the tiled version.  The K loop is broken into TILE-wide chunks.
// For each chunk every thread cooperatively loads ONE element of A and ONE of
// B into shared memory; after a __syncthreads() the whole block multiplies the
// two cached tiles, reusing each loaded float TILE times.  That cuts global
// memory traffic by ~TILE and is what makes GPUs fast at matmul.
//
// This is the kernel we dump to PTX and SASS -- watch for:
//   * .shared memory declarations (As/Bs below)
//   * bar.sync             (the __syncthreads barriers)
//   * fma.rn.f32           (the fused multiply-add doing the actual math)
//   * ld.shared / ld.global (fast on-chip vs slow off-chip loads)
// ---------------------------------------------------------------------------
__global__ void sgemm_tiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];                         // a tile of A, on-chip
    __shared__ float Bs[TILE][TILE];                         // a tile of B, on-chip

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    // slide a TILE-wide window along the shared K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();                                     // tile fully loaded

        #pragma unroll
        for (int k = 0; k < TILE; ++k)                       // multiply the cached tiles
            acc += As[ty][k] * Bs[k][tx];                    // ld.shared, ld.shared, fma
        __syncthreads();                                     // done before reloading
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// CPU reference + a tiny benchmark harness.
// ---------------------------------------------------------------------------
static void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) { float d = fabsf(a[i] - b[i]); if (d > m) m = d; }
    return m;
}

int main() {
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (sm_%d%d, %d SMs, %d-bit mem bus)\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount, prop.memoryBusWidth);

    // A 512x512 multiply -- the kind of GEMM hiding inside an attention layer.
    int M = 512, N = 512, K = 512;
    size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
    float *hA = (float*)malloc(szA * 4), *hB = (float*)malloc(szB * 4);
    float *hC = (float*)malloc(szC * 4), *hRef = (float*)malloc(szC * 4);
    for (size_t i = 0; i < szA; ++i) hA[i] = (float)rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < szB; ++i) hB[i] = (float)rand() / RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, szA * 4)); CUDA_CHECK(cudaMalloc(&dB, szB * 4));
    CUDA_CHECK(cudaMalloc(&dC, szC * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA, szA * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, szB * 4, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cpu_matmul(hA, hB, hRef, M, N, K);                       // ground truth

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    const char* names[2] = {"naive", "tiled"};
    for (int which = 0; which < 2; ++which) {
        // warm up + correctness
        if (which == 0) sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
        else            sgemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hC, dC, szC * 4, cudaMemcpyDeviceToHost));
        float err = max_abs_diff(hC, hRef, szC);

        int iters = 50;                                      // timed loop
        cudaEventRecord(t0);
        for (int it = 0; it < iters; ++it) {
            if (which == 0) sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
            else            sgemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
        }
        cudaEventRecord(t1); CUDA_CHECK(cudaEventSynchronize(t1));
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= iters;
        double gflops = 2.0 * M * N * K / (ms * 1e6);
        printf("%-6s  max|err|=%.2e   %.3f ms   %.0f GFLOP/s\n",
               names[which], err, ms, gflops);
    }
    printf("(both kernels match the CPU reference; tiled reuses shared memory)\n");
    free(hA); free(hB); free(hC); free(hRef);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
