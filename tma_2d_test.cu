#include <cuda.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

#define CU_CHECK(err) do { \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        std::cerr << "Driver API error: " << errStr << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// Dimensions
constexpr int GMEM_WIDTH = 16;
constexpr int GMEM_HEIGHT = 16;
constexpr int SMEM_WIDTH = 16;
constexpr int SMEM_HEIGHT = 16;

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    void* func_ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &func_ptr, 12000, cudaEnableDefault, &status));
    assert(status == cudaDriverEntryPointSuccess);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(func_ptr);
}

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
    __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token token;
    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x, y, bar);
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
    } else {
        token = bar.arrive();
    }

    bar.wait(std::move(token));

    smem_buffer[0][threadIdx.x] += threadIdx.x;

    cde::fence_proxy_async_shared_cta();
    __syncthreads();

    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, &smem_buffer);
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
    }

    if (threadIdx.x == 0) {
        (&bar)->~barrier();
    }
}

int main() {
    CUtensorMap tensor_map{};
    PFN_cuTensorMapEncodeTiled_v12000 encode = get_cuTensorMapEncodeTiled();

    constexpr int rank = 2;
    uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
    uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
    uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
    uint32_t elem_stride[rank] = {1, 1};

    // Allocate input/output buffer
    int* d_tensor;
    CUDA_CHECK(cudaMalloc(&d_tensor, GMEM_WIDTH * GMEM_HEIGHT * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tensor, 0, GMEM_WIDTH * GMEM_HEIGHT * sizeof(int)));

    CU_CHECK(encode(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank,
        d_tensor,
        size,
        stride,
        box_size,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    kernel<<<1, SMEM_WIDTH>>>(tensor_map, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Readback for sanity
    int h_tensor[GMEM_WIDTH * GMEM_HEIGHT];
    CUDA_CHECK(cudaMemcpy(h_tensor, d_tensor, sizeof(h_tensor), cudaMemcpyDeviceToHost));

    std::cout << "Post-kernel tensor[0..15]: ";
    for (int i = 0; i < GMEM_WIDTH; ++i) {
        std::cout << h_tensor[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_tensor);
    return 0;
}
