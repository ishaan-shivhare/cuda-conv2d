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

    uint64_t input_dims[RANK] = {static_cast<uint64_t>(IN_C), static_cast<uint64_t>(H), static_cast<uint64_t>(W)};
    uint64_t input_strides[RANK - 1] = {
        static_cast<uint64_t>(H * W * sizeof(float)),
        static_cast<uint64_t>(W * sizeof(float))
    };
    uint32_t input_box[RANK] = {4, static_cast<uint32_t>(SH_TILE_W), static_cast<uint32_t>(SH_TILE_W)};
    uint32_t input_elem_stride[RANK] = {1, 1, 1};

    uint64_t output_dims[RANK_OUT] = {static_cast<uint64_t>(padded_out_h), static_cast<uint64_t>(padded_out_w)};
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


// __device__ void producer(const CUtensorMap& tensor_map, barrier ready[], barrier filled[], float* buf0, float* buf1)
// {
    
//     printf("In producer: block (%d, %d)\n", blockIdx.x, blockIdx.y);
//     printf("Tile coords: row = %d, col = %d\n", blockIdx.y * BLOCK_SIZE, blockIdx.x * BLOCK_SIZE);
//     for (int c = 0; c < IN_C; c+=4) {
//         int buf = (c / 4) % 2;
//         float* tile_ptr = (buf == 0) ? buf0 : buf1;
//         ready[buf].arrive_and_wait(); /* wait for buffer_(buf) to be ready to be filled */
//         /* produce, i.e., fill in, buffer_(buf)  */
//         printf("TMA about to launch: block (%d, %d), c = %d\n", blockIdx.x, blockIdx.y, c);
//         // printf("Shared buffer address (buf=%d): %p\n", buf, reinterpret_cast<void*>(&smem_buffer[buf]));
//         cde::cp_async_bulk_tensor_3d_global_to_shared(reinterpret_cast<void*>(tile_ptr), &tensor_map, c, blockIdx.y * BLOCK_SIZE, blockIdx.x * BLOCK_SIZE, filled[buf]);
//         printf("TMA issued successfully for block (%d, %d)\n", blockIdx.x, blockIdx.y);
//         barrier::arrival_token token = cuda::device::barrier_arrive_tx(filled[buf], 1, 4 * SH_TILE_W * SH_TILE_W * sizeof(float)); /* buffer_(buf) is filled */
//         filled[buf].wait(std::move(token));
//     }
// }

// __device__ void consumer(barrier ready[], barrier filled[], float* buf0, float* buf1, float smem_out[BLOCK_SIZE][BLOCK_SIZE], float smem_kernel[IN_C][K][K], int OUT_H, int OUT_W)
// {
//     if (threadIdx.z == 1 && threadIdx.x == 1 && threadIdx.y == 1)
//             printf("In consumer of block (%d, %d)\n", blockIdx.x, blockIdx.y);
       
//     // reinterpret each aligned buffer into [4][SH_TILE_W][SH_TILE_W]
//     float (*b0)[SH_TILE_W][SH_TILE_W] = reinterpret_cast<float(*)[SH_TILE_W][SH_TILE_W]>(buf0);
//     float (*b1)[SH_TILE_W][SH_TILE_W] = reinterpret_cast<float(*)[SH_TILE_W][SH_TILE_W]>(buf1);

//     barrier::arrival_token token1 = ready[0].arrive();
//     barrier::arrival_token token2 = ready[1].arrive();

//     int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
//     if (tid == 0) return;

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int tz = threadIdx.z;

//     int global_y = blockIdx.y * BLOCK_SIZE + ty;
//     int global_x = blockIdx.x * BLOCK_SIZE + tx;

//     for (int c = 0; c < IN_C; c += 4) {
//         int buf = (c / 4) % 2;

//         // filled[buf].arrive_and_wait();
//         barrier::arrival_token token = filled[buf].arrive();
//         // Wait for all threads (and for the TMA engine) to finish the load
//         filled[buf].wait(std::move(token));
//         if (threadIdx.z == 1 && threadIdx.x == 1 && threadIdx.y == 1)
//             printf("Computing on buffer (%d) for block (%d, %d)\n", buf, blockIdx.x, blockIdx.y);
        
//         if (blockIdx.x==0 && tz==0 && ty<2 && tx<2) {
//             float v = (buf==0) ? b0[0][ty][tx] : b1[0][ty][tx];
//             printf("DBG load buf=%d [%d,%d]=%f\n", buf, ty, tx, v);
//         }

//         if (tz < 4 && global_y < OUT_H && global_x < OUT_W && (c + tz) < IN_C) {
//             float accum = 0.0f;
//             for (int ky = 0; ky < K; ++ky) {
//                 for (int kx = 0; kx < K; ++kx) {
//                     int shared_y = ty + ky;
//                     int shared_x = tx + kx;
//                     if (shared_y < SH_TILE_W && shared_x < SH_TILE_W) {
//                         float val = (buf == 0) ?  b0[tz][ty + ky][tx + kx] : b1[tz][ty + ky][tx + kx];
//                         float weight = smem_kernel[c + tz][ky][kx];
//                         accum += val * weight;
//                     }
//                 }
//             }

//             // Accumulate into shared output
//             atomicAdd(&smem_out[ty][tx], accum);
//         }

        
//         ready[buf].arrive();  // signal buffer is free again
//         if (threadIdx.z == 1 && threadIdx.x == 1 && threadIdx.y == 1)
//             printf("Finished a computation on block (%d, %d)\n", blockIdx.x, blockIdx.y);

//     }
// }


// //N is the total number of float elements in arrays in and out
// __global__ void producer_consumer_pattern(const __grid_constant__ CUtensorMap input_map, const __grid_constant__ CUtensorMap output_map, float* kernel, float* out, int OUT_H, int OUT_W) {

//     if (threadIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0)
//         printf("Launching producer-consumer kernel on block (%d, %d)\n", blockIdx.x, blockIdx.y);

//     // Shared memory buffer declared below is of size 2 * buffer_len
//     // so that we can alternatively work between two buffers.
//     // buffer_0 = buffer and buffer_1 = buffer + buffer_len
//     // __shared__ extern float buffer[];
//     // __shared__ alignas(128) float smem_raw[2 * 4 * SH_TILE_W * SH_TILE_W];
//     // float (*smem_buffer)[4][SH_TILE_W][SH_TILE_W] = reinterpret_cast<float(*)[4][SH_TILE_W][SH_TILE_W]>(smem_raw);
//     // two 128‑byte aligned SMEM tiles of shape [4][SH_TILE_W][SH_TILE_W]
//     __shared__ alignas(128) float smem_buf0[4][SH_TILE_W][SH_TILE_W];
//     __shared__ alignas(128) float smem_buf1[4][SH_TILE_W][SH_TILE_W];


//     __shared__ alignas(128) float smem_out[BLOCK_SIZE][BLOCK_SIZE];

//     __shared__ float smem_kernel[IN_C][K][K];

//     if (threadIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0)
//         printf("declared shared mem on block (%d, %d)\n", blockIdx.x, blockIdx.y);

//     // bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
//     // while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively
//     __shared__ barrier bar[4];
    
//     auto block = cooperative_groups::this_thread_block();
//     if (block.thread_rank() < 4){
//             init(bar + block.thread_rank(), block.size());
//             cde::fence_proxy_async_shared_cta();  // make visible to async agent
//     } 
//     // block.sync();
//     int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
//     int total_threads = blockDim.z * blockDim.y * blockDim.x;
//     // Load kernel weights to smem
//     int total_weights = IN_C * K * K;
//     for (int i = tid; i < total_weights; i += total_threads) {
//         int c = i / (K * K);
//         int k = i % (K * K);
//         int ky = k / K;
//         int kx = k % K;
//         smem_kernel[c][ky][kx] = kernel[c * K * K + ky * K + kx];
//     }

//     // Zero out smem_out — only z==0 threads do this
//     if (threadIdx.z == 0) {
//         for (int i = threadIdx.y * blockDim.x + threadIdx.x;
//             i < BLOCK_SIZE * BLOCK_SIZE;
//             i += blockDim.x * blockDim.y) {

//             int y = i / BLOCK_SIZE;
//             int x = i % BLOCK_SIZE;
//             smem_out[y][x] = 0.0f;
//         }
//     }
//     __syncthreads();
    
//     if (threadIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0)
//         printf("Initialised barriers, loaded kernel, zeroed output tile on block (%d, %d)\n", blockIdx.x, blockIdx.y);


//     if (tid == 0) {
//         printf("Calling producer on block (%d, %d)\n", blockIdx.x, blockIdx.y);
//         producer(input_map, bar, bar+2, &smem_buf0[0][0][0], &smem_buf1[0][0][0]);
//     } else {
//         if (threadIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0)
//             printf("Entering consumer on block (%d, %d)\n", blockIdx.x, blockIdx.y);
//         consumer(bar, bar+2, &smem_buf0[0][0][0], &smem_buf1[0][0][0], smem_out, smem_kernel, OUT_H, OUT_W);

//     }
//     __syncthreads();
//     if (tid == 0) {
//         printf("Committing output tile for block (%d, %d)\n", blockIdx.x, blockIdx.y);
//         cde::cp_async_bulk_tensor_2d_shared_to_global(&output_map, blockIdx.y * BLOCK_SIZE, blockIdx.x * BLOCK_SIZE, reinterpret_cast<void*>(&smem_out[0][0]));
//         cde::cp_async_bulk_commit_group();
//         cde::cp_async_bulk_wait_group_read<0>();
//     }

// }

// N.B. you can delete your old producer() and consumer() helpers entirely
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
    __shared__ alignas(128) float smem_buf0[4][SH_TILE_W][SH_TILE_W];
    __shared__ alignas(128) float smem_buf1[4][SH_TILE_W][SH_TILE_W];
    __shared__ alignas(128) float smem_out[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float          smem_kernel[IN_C][K][K];
    __shared__ barrier        bar_filled[2];

    // Thread index helpers
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = tz*blockDim.y*blockDim.x + ty*blockDim.x + tx;
    int lane_count = blockDim.x*blockDim.y*blockDim.z;

    // Init the “filled” barriers once
    if (tid == 0) {
        init(&bar_filled[0], lane_count);
        init(&bar_filled[1], lane_count);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

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
    if (tz == 0) {
        for (int i = ty*blockDim.x + tx; i < BLOCK_SIZE*BLOCK_SIZE; i += blockDim.x*blockDim.y) {
            int y = i / BLOCK_SIZE, x = i % BLOCK_SIZE;
            smem_out[y][x] = 0.0f;
        }
    }
    __syncthreads();

    if (tid == 0) {
        printf("Initialised barriers, loaded kernel, zeroed output tile on block (%d, %d)\n", bx, by);
    }


    if (tid == 0) {
        int H = 16;
        int W = 16;
        float* g = reinterpret_cast<float*>(inp);
        // print the very first two rows of channel‑0:
        printf("[GMEM ch0 row0] %f %f %f %f\n",
                g[0*H*W + 0*W + 0],
                g[0*H*W + 0*W + 1],
                g[0*H*W + 0*W + 2],
                g[0*H*W + 0*W + 3]);
        printf("[GMEM ch0 row1] %f %f %f %f\n",
                g[0*H*W + 1*W + 0],
                g[0*H*W + 1*W + 1],
                g[0*H*W + 1*W + 2],
                g[0*H*W + 1*W + 3]);
    }


    // Pre‑load chunk 0 into buf 0
    int buf      = 0;
    int c        = 0;
    barrier::arrival_token tok[2];
    if (tid == 0) {
        printf("In producer preload: block (%d,%d), c=%d\n", bx, by, c);
        cde::cp_async_bulk_tensor_3d_global_to_shared(
            &smem_buf0[0][0][0],
            &input_map,
            c,
            by*BLOCK_SIZE,
            bx*BLOCK_SIZE,
            bar_filled[0]
        );
        tok[0] = cuda::device::barrier_arrive_tx(
            bar_filled[0], 1,
            4ULL * SH_TILE_W * SH_TILE_W * sizeof(float)
        );
    }
    else {
        tok[0] = bar_filled[0].arrive();
    }
    // __syncthreads();
    // wait + flush buf0
    bar_filled[0].wait(std::move(tok[0]));
    cde::fence_proxy_async_shared_cta();
    __syncthreads();

    if (tid == 0) {
        // after bar_filled[0].wait(...)
        for (int c = 0; c < 4; ++c) {
            printf("[Preload ch %d] buf0[%d][0][0]=%f   [%d][0][1]=%f   [%d][1][0]=%f\n",
                c,
                c, smem_buf0[c][0][0],
                c, smem_buf0[c][0][1],
                c, smem_buf0[c][1][0]);
        }
    }

    // Main pipeline loop
    for (int c = 0; c < IN_C; c += 4) {
        int next_c   = c + 4;
        int next_buf = buf ^ 1;
        float (*cur_buf)[SH_TILE_W][SH_TILE_W] = (buf==0)
            ? smem_buf0
            : smem_buf1;

        
        if (tid == 0) {
            printf("[Loop start c=%d] cur_buf=%d sample smem_buf%d[0][0][0]=%f, [0][1][1]=%f\n",
                    c, buf,
                    buf,
                    ((buf==0)? smem_buf0[0][0][0] : smem_buf1[0][0][0]),
                    ((buf==0)? smem_buf0[0][1][1] : smem_buf1[0][1][1])
            );
        }
        __syncthreads();
        // 1) Kick off prefetch of next chunk into next_buf
        if (next_c < IN_C && tid == 0) {
            printf("In producer: block (%d,%d), c=%d\n", bx, by, next_c);
            cde::cp_async_bulk_tensor_3d_global_to_shared(
                &((next_buf==0? smem_buf0: smem_buf1)[0][0][0]),
                &input_map,
                next_c,
                by*BLOCK_SIZE,
                bx*BLOCK_SIZE,
                bar_filled[next_buf]
            );
            tok[next_buf] = cuda::device::barrier_arrive_tx(
                bar_filled[next_buf], 1,
                4ULL*SH_TILE_W*SH_TILE_W*sizeof(float)
            );
        } else {
            tok[next_buf] = bar_filled[next_buf].arrive();
        }
        

        // 2) Compute on cur_buf
        if (tz==1 && ty==1 && tx==1)
            printf("Computing on buffer (%d) for block (%d,%d)\n", buf, bx, by);
        int global_y = by*BLOCK_SIZE + ty;
        int global_x = bx*BLOCK_SIZE + tx;
        if (tz < 4 && global_y < OUT_H && global_x < OUT_W && (c+tz) < IN_C) {
            float accum = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    int sy = ty+ky, sx = tx+kx;
                    if (sy < SH_TILE_W && sx < SH_TILE_W) {
                        float val    = cur_buf[tz][sy][sx];
                        float weight = smem_kernel[c+tz][ky][kx];
                        accum += val * weight;
                    }
                }
            }
            atomicAdd(&smem_out[ty][tx], accum);
        }
        

        // 3) Wait + flush next_buf (so it's ready for next iteration)
        if (next_c < IN_C) {
            // __syncthreads();
            bar_filled[next_buf].wait(std::move(tok[next_buf]));
            cde::fence_proxy_async_shared_cta();
            __syncthreads();
        }

        buf = next_buf;
    }

    __syncthreads();
    cde::fence_proxy_async_shared_cta();
    // Final TMA store of smem_out → global
    if (tid == 0) {
        printf("Committing output tile for block (%d, %d)\n", bx, by);
        cde::cp_async_bulk_tensor_2d_shared_to_global(
            &output_map,
            by*BLOCK_SIZE,
            bx*BLOCK_SIZE,
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

    // TORCH_CHECK(output.size(0) == OUT_H && output.size(1) == OUT_W, "Output shape mismatch");
    // TORCH_CHECK(output.dim() == 3 &&
    //         output.size(0) == 1 &&
    //         output.size(1) == OUT_H &&
    //         output.size(2) == OUT_W,
    //         "Output shape must be [1, H_out, W_out]");

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
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 4);  // z-dim needed due to TMA rules
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
