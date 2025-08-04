#include "conv2d_config.h" 
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled
#include <torch/extension.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

#define CUDA_CHECK(err) do { \
    cudaError_t err_ = err; \
    if (err_ != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

#define CU_CHECK(err) do { \
    CUresult err_ = err; \
    if (err_ != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err_, &errStr); \
        printf("Driver API error: %s at %s:%d\n", errStr, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)


PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    void* func_ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &func_ptr, 12000, cudaEnableDefault, &status));
    assert(status == cudaDriverEntryPointSuccess);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(func_ptr);
}

// Wraps input and output CUtensorMap creation
void setup_tensor_maps(CUtensorMap& input_map, CUtensorMap& output_map,
                       void* d_input, void* d_output,
                       int IN_C, int H, int W,
                       int padded_out_h, int padded_out_w,
                       int BLOCK_SIZE) {
    constexpr uint32_t RANK = 3;
    constexpr uint32_t RANK_OUT = 2;

    uint64_t input_dims[RANK] = {static_cast<uint64_t>(W), static_cast<uint64_t>(H), static_cast<uint64_t>(IN_C)};
    uint64_t input_strides[RANK - 1] = {
        static_cast<uint64_t>(W * sizeof(float)),
        static_cast<uint64_t>(H * W * sizeof(float))
    };
    uint32_t input_box[RANK] = {static_cast<uint32_t>(TILE_W_PAD), static_cast<uint32_t>(SH_TILE_W), 1};
    uint32_t input_elem_stride[RANK] = {1, 1, 1};

    uint64_t output_dims[RANK_OUT] = {static_cast<uint64_t>(padded_out_w), static_cast<uint64_t>(padded_out_h)};
    uint64_t output_strides[RANK_OUT - 1] = {
        static_cast<uint64_t>(padded_out_w* sizeof(float))
    };
    uint32_t output_box[RANK_OUT] = {static_cast<uint32_t>(BLOCK_SIZE), static_cast<uint32_t>(BLOCK_SIZE)};
    uint32_t output_elem_stride[RANK_OUT] = {1, 1};

    auto encode = get_cuTensorMapEncodeTiled();

    CUresult res = encode(
        &input_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        RANK,
        d_input,
        input_dims,
        input_strides,
        input_box,
        input_elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    // assert(res == CUDA_SUCCESS);
    if (res != CUDA_SUCCESS) {
        printf("cuTensorMapEncodeTiled for input_map failed with error code: %d\n", res);
        std::abort();
    }


    res = encode(
        &output_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        RANK_OUT,
        d_output,
        output_dims,
        output_strides,
        output_box,
        output_elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res != CUDA_SUCCESS) {
        printf("cuTensorMapEncodeTiled for output_map failed with error code: %d\n", res);
        std::abort();
    }
}

__global__ void producer_consumer_pattern(
    const __grid_constant__ CUtensorMap input_map,
    const __grid_constant__ CUtensorMap output_map,
    float* kernel,
    float* out,
    int OUT_H,
    int OUT_W,
    float* inp,
    int H,
    int W,
    int padded_out_w
) {

    // Shared memory buffer declared below is of size 2 * buffer_len
    // so that we can alternatively work between two buffers.
    // TODO: Maybe merge into one big buffer of size some constant instead of separate 
    // Shared tiles
    __shared__ alignas(128) float smem_buf0[SH_TILE_W][TILE_W_PAD];
    __shared__ alignas(128) float smem_buf1[SH_TILE_W][TILE_W_PAD];
    __shared__ alignas(128) float smem_out[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float smem_kernel[IN_C][K][K];
    __shared__ barrier bar_ready[2]; // track if buffers buffer_0 and buffer_1 are ready to be filled,
    __shared__ barrier bar_filled[2]; // track if buffers buffer_0 and buffer_1 are filled-in respectively

    int tid = threadIdx.y*blockDim.x + threadIdx.x;
    if (tid < 2) {
        init(bar_ready + tid, blockDim.x * blockDim.y);
        init(bar_filled + tid, blockDim.x * blockDim.y);
        cde::fence_proxy_async_shared_cta();
    }
    
    // Load kernel into SMEM
    for (int i = tid; i < IN_C * K * K; i += blockDim.x * blockDim.y) {
        int c  = i / (K*K);
        int kk = i % (K*K);
        int ky = kk / K, kx = kk % K;
        smem_kernel[c][ky][kx] = kernel[c*K*K + ky*K + kx];
    }
    
    // Zero out smem_out
    for (int i = tid; i < BLOCK_SIZE*BLOCK_SIZE; i += blockDim.x * blockDim.y) {
        int y = i / BLOCK_SIZE, x = i % BLOCK_SIZE;
        smem_out[y][x] = 0.0f;
    }
    
    // Sync to make barriers, kernel, zeroed out accumulator visible to everyone
    __syncthreads();

    // For now: 1 producer warpgroup and 1 consumer warpgroup (16x16 block)
    // Can increase block size to 16x32 for 3 consumer 1 producer
    if (tid < warpSize * 4) {
        // consumer
        barrier::arrival_token token1 = bar_ready[0].arrive(); // buffer 0 is ready for initial fill 
        barrier::arrival_token token2 = bar_ready[1].arrive(); // buffer 1 is ready for initial fill 
        for (int c = 0; c < IN_C; ++c) {
            bar_filled[c%2].arrive_and_wait();
            float (*cur_buf)[TILE_W_PAD] = (c%2==0) ? smem_buf0 : smem_buf1;
            for (int i = tid; i < BLOCK_SIZE*BLOCK_SIZE; i += warpSize * 4) {
                int local_y = i / BLOCK_SIZE;
                int local_x = i % BLOCK_SIZE;
                float accum = 0.0f;
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        int sy = local_y + ky;
                        int sx = local_x + kx;
                        if (sy < SH_TILE_W && sx < SH_TILE_W) {
                            accum += cur_buf[sy][sx] * smem_kernel[c][ky][kx];
                        }
                    }
                }
                smem_out[local_y][local_x] += accum;
            }
            barrier::arrival_token token = bar_ready[c%2].arrive();
        }

    }
    else {
        // producer
        // TODO: give up registers with "setmaxnreg" inline PTX 
        for (int c = 0; c < IN_C; ++c) {
            // Determine which buffer to load into
            float (*cur_buf)[TILE_W_PAD] = (c%2==0) ? smem_buf0 : smem_buf1;
            bar_ready[c%2].arrive_and_wait();
            barrier::arrival_token t_load;
            if (tid == 128) {
                // printf("reading channel %d from global for block (%d, %d) \n", c, blockIdx.x, blockIdx.y);
                cde::cp_async_bulk_tensor_3d_global_to_shared(
                    &cur_buf[0][0],
                    &input_map,
                    blockIdx.x*BLOCK_SIZE,
                    blockIdx.y*BLOCK_SIZE,
                    c,
                    bar_filled[c%2]
                );
                t_load = cuda::device::barrier_arrive_tx(bar_filled[c%2], 1, SH_TILE_W * TILE_W_PAD * sizeof(float));
            }
            else {
                t_load = bar_filled[c%2].arrive();
            }
        }
    }
    
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // Final TMA store of smem_out → global
    if (tid == 0) {
        // printf("Committing output tile for block (%d, %d)\n", blockIdx.x, blockIdx.y);
        cde::cp_async_bulk_tensor_2d_shared_to_global(
            &output_map,
            blockIdx.x*BLOCK_SIZE,
            blockIdx.y*BLOCK_SIZE,
            &smem_out[0][0]
        );
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
    }
}

void launch_conv2d_tma(torch::Tensor input, torch::Tensor kernel, torch::Tensor output, int padded_out_h, int padded_out_w) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(kernel.device().is_cuda(), "kernel must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "kernel must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");

    TORCH_CHECK(input.dim() == 3, "input must be [C, H, W]");
    TORCH_CHECK(kernel.dim() == 3, "kernel must be [C, K, K]");
    TORCH_CHECK(output.dim() == 2, "output must be [H_out, W_out]");

    TORCH_CHECK(input.size(0) == IN_C, "IN_C mismatch");
    TORCH_CHECK(kernel.size(0) == IN_C, "IN_C mismatch");
    TORCH_CHECK(kernel.size(1) == K && kernel.size(2) == K, "Kernel shape mismatch");

    int H = input.size(1);
    int W = input.size(2);
    int OUT_H = H - K + 1;
    int OUT_W = W - K + 1;

    // Setup tensor maps
    CUtensorMap input_map;
    CUtensorMap output_map;
    setup_tensor_maps(
        input_map, output_map,
        input.data_ptr<float>(), output.data_ptr<float>(),
        IN_C, H, W,
        padded_out_h, padded_out_w,
        BLOCK_SIZE
    );

    // Kernel launch
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);  // z-dim needed due to TMA rules
    dim3 blocks(
        // (OUT_W + BLOCK_SIZE - 1) / BLOCK_SIZE,
        // (OUT_H + BLOCK_SIZE - 1) / BLOCK_SIZE
        H / BLOCK_SIZE, W / BLOCK_SIZE
    );

    producer_consumer_pattern<<<blocks, threads>>>(
        input_map,
        output_map,
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        OUT_H,
        OUT_W,
        input.data_ptr<float>(),
        H,
        W,
        padded_out_w
    );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel or TMA error: " << cudaGetErrorString(err) << std::endl;
    }
}
