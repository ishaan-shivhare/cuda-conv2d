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
    float* inp
) {
    // Shared tiles
    __shared__ alignas(128) float smem_buf0[SH_TILE_W][TILE_W_PAD];
    __shared__ alignas(128) float smem_buf1[SH_TILE_W][TILE_W_PAD];
    __shared__ alignas(128) float smem_out[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float          smem_kernel[IN_C][K][K];
    __shared__ barrier        bar_filled[2];

    // Thread index helpers
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = ty*blockDim.x + tx;
    int lane_count = blockDim.x*blockDim.y;

    // Init the “filled” barriers once
    if (tid == 0) {
        init(&bar_filled[0], lane_count);
        init(&bar_filled[1], lane_count);
        cde::fence_proxy_async_shared_cta();
    }

    // Load kernel into SMEM
    int total_threads = lane_count;
    int total_weights = IN_C * K * K;
    for (int i = tid; i < total_weights; i += total_threads) {
        int c  = i / (K*K);
        int kk = i % (K*K);
        int ky = kk / K, kx = kk % K;
        smem_kernel[c][ky][kx] = kernel[c*K*K + ky*K + kx];
    }

    // Zero out smem_out
    
    for (int i = ty*blockDim.x + tx; i < BLOCK_SIZE*BLOCK_SIZE; i += blockDim.x*blockDim.y) {
        int y = i / BLOCK_SIZE, x = i % BLOCK_SIZE;
        smem_out[y][x] = 0.0f;
    }
    
    __syncthreads();

    // Pre‑load chunk 0 into buf 0
    int c = 0;
    barrier::arrival_token t0;
    if (tid == 0) {
        // printf("In producer preload: block (%d,%d), c=%d\n", bx, by, c);
        cde::cp_async_bulk_tensor_3d_global_to_shared(
            &smem_buf0[0][0],
            &input_map,
            bx*BLOCK_SIZE,
            by*BLOCK_SIZE,
            c,
            bar_filled[0]
        );
        t0 = cuda::device::barrier_arrive_tx(
            bar_filled[0], 1,
            SH_TILE_W * TILE_W_PAD * sizeof(float)
        );
    }
    else {
        t0 = bar_filled[0].arrive();
    }

    // wait + flush buf0
    bar_filled[0].wait(std::move(t0));


    int buf = 0;
    // Main pipeline loop
    for (int c = 0; c < IN_C; ++c) {
        
        // Prefetch next
        int next_c = c + 1;
        int next_buf = buf ^ 1;
        barrier::arrival_token tn;        

        // 1) Kick off prefetch of next chunk into next_buf
        float (*cur_buf)[TILE_W_PAD] = (buf==0) ? smem_buf0 : smem_buf1;
        
        if (next_c < IN_C){
            if (tid == 0) {
                cde::cp_async_bulk_tensor_3d_global_to_shared(
                    &((next_buf==0 ? smem_buf0 : smem_buf1))[0][0],
                    &input_map,
                    bx*BLOCK_SIZE,
                    by*BLOCK_SIZE,
                    next_c,
                    bar_filled[next_buf]
                );
                tn = cuda::device::barrier_arrive_tx(
                        bar_filled[next_buf], 1,
                        SH_TILE_W * TILE_W_PAD * sizeof(float)
                );
            } else {
                tn = bar_filled[next_buf].arrive();
            }
        }
        
        int global_y = by*BLOCK_SIZE + ty;
        int global_x = bx*BLOCK_SIZE + tx;
        if (global_y < OUT_H && global_x < OUT_W) {
            float accum = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    int sy = ty + ky; 
                    int sx = tx + kx;
                    if (sy < SH_TILE_W && sx < SH_TILE_W) {
                        float val    = cur_buf[sy][sx];
                        float weight = smem_kernel[c][ky][kx];
                        accum += val * weight;
                    }
                }
            }
            smem_out[ty][tx] += accum;
        }

        // 3) Wait + flush next_buf (so it's ready for next iteration)
        if (next_c < IN_C) {
            bar_filled[next_buf].wait(std::move(tn));
        }

        buf = next_buf;
    }

    
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // Final TMA store of smem_out → global
    if (tid == 0) {
        // printf("Committing output tile for block (%d, %d)\n", bx, by);
        cde::cp_async_bulk_tensor_2d_shared_to_global(
            &output_map,
            bx*BLOCK_SIZE,
            by*BLOCK_SIZE,
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
        input.data_ptr<float>()
    );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel or TMA error: " << cudaGetErrorString(err) << std::endl;
    }
}
