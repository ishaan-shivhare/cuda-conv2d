#include <cuda.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled, CUtensorMap

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

constexpr int C = 4;
constexpr int H = 4;
constexpr int W = 4;

__global__ void tma_3d_kernel(const __grid_constant__ CUtensorMap tensor_map) {
    __shared__ alignas(128) int smem_buffer[C][H][W];
    __shared__ barrier bar;

    // Thread 0 initializes the barrier
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        cde::fence_proxy_async_shared_cta();  // make visible to async agent
    }

    __syncthreads(); // ensure all threads see initialized barrier

    barrier::arrival_token token;
    if (threadIdx.x == 0) {
        // Start the TMA transfer after barrier is fully visible
        cde::cp_async_bulk_tensor_3d_global_to_shared(&smem_buffer, &tensor_map, 0, 0, 0, bar);
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
    } else {
        token = bar.arrive();
    }

    bar.wait(std::move(token));

    // Safe 3D indexing
    int t = threadIdx.x;
    if (t < C * H * W) {
        int c = t / (H * W);
        int h = (t / W) % H;
        int w = t % W;
        smem_buffer[c][h][w] += 1;
    }

    cde::fence_proxy_async_shared_cta();
    __syncthreads();

    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_3d_shared_to_global(&tensor_map, 0, 0, 0, &smem_buffer);
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
        (&bar)->~barrier();
    }
}

PFN_cuTensorMapEncodeTiled_v12000 get_encoder() {
    void* func_ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &func_ptr, 12000, cudaEnableDefault, &status));
    assert(status == cudaDriverEntryPointSuccess);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(func_ptr);
}

int main() {
    constexpr int numel = C * H * W;
    int* d_tensor;
    CUDA_CHECK(cudaMalloc(&d_tensor, numel * sizeof(int)));

    int h_tensor[numel];
    for (int i = 0; i < numel; ++i)
        h_tensor[i] = i;

    CUDA_CHECK(cudaMemcpy(d_tensor, h_tensor, numel * sizeof(int), cudaMemcpyHostToDevice));

    CUtensorMap tensor_map{};
    PFN_cuTensorMapEncodeTiled_v12000 encoder = get_encoder();

    uint64_t dims[3] = {C, H, W};
    uint64_t strides[2] = {H * W * sizeof(int), W * sizeof(int)};
    uint32_t box[3] = {C, H, W};
    uint32_t elem_stride[3] = {1, 1, 1};

    CU_CHECK(encoder(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        3,
        d_tensor,
        dims,
        strides,
        box,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    tma_3d_kernel<<<1, C * H * W>>>(tensor_map);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_result[numel];
    CUDA_CHECK(cudaMemcpy(h_result, d_tensor, sizeof(h_result), cudaMemcpyDeviceToHost));

    std::cout << "Post-kernel tensor:\n";
    for (int i = 0; i < numel; ++i) {
        std::cout << h_result[i] << " ";
        if ((i + 1) % W == 0) std::cout << "\n";
        if ((i + 1) % (H * W) == 0) std::cout << "\n";
    }

    CUDA_CHECK(cudaFree(d_tensor));
    return 0;
}
