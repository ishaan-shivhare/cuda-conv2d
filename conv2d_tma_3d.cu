#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled
#include <torch/extension.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// --------------------------- Tunables ---------------------------
constexpr int BLOCK_SIZE  = 16; // logical output tile size per CTA (spatial dimension)
constexpr int BLOCK_DEPTH = 16; // number of output channels per CTA
constexpr int DEPTH       = 3; // input ring pipeline depth
constexpr int K           = 3;
constexpr int IN_C        = 32;
constexpr int OUT_C       = 128;

// Warpgroup specialization (Hopper): 1 warpgroup = 4 warps = 128 threads
constexpr int WARP_SZ      = 32;
constexpr int WARPS_PER_WG = 4;
constexpr int WG_SIZE      = WARPS_PER_WG * WARP_SZ; // 128

// Choose #consumer & #producer warpgroups (decoupled from BLOCK_SIZE)
constexpr int CONSUMER_WGS = 2;  // 1 consumer warpgroup (128 threads)
constexpr int PRODUCER_WGS = 1;  // 1 producer warpgroup (128 threads)

// Derived
constexpr int NUM_CONSUMERS   = CONSUMER_WGS * WG_SIZE;       // 128 threads
constexpr int NUM_PRODUCERS   = PRODUCER_WGS * WG_SIZE;       // 128 threads
constexpr int THREADS_PER_CTA = NUM_CONSUMERS + NUM_PRODUCERS; // 256

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
                       int N, int IN_C, int H, int W,
                       int padded_out_h, int padded_out_w,
                       int C_OUT_PAD,
                       int sh_tile_h, int tile_w_pad,
                       int BLOCK_SIZE) {
    
    constexpr uint32_t RANK = 3, RANK_OUT = 3;

    uint64_t input_dims[RANK] = {static_cast<uint64_t>(W), static_cast<uint64_t>(H), static_cast<uint64_t>(N * IN_C)};
    uint64_t input_strides[RANK - 1] = {
        static_cast<uint64_t>(W * sizeof(float)),
        static_cast<uint64_t>(H * W * sizeof(float))
    };
    uint32_t input_box[RANK] = {static_cast<uint32_t>(tile_w_pad), static_cast<uint32_t>(sh_tile_h), 1};
    uint32_t input_elem_stride[RANK] = {1, 1, 1};

    uint64_t output_dims[RANK_OUT] = {static_cast<uint64_t>(padded_out_w), static_cast<uint64_t>(padded_out_h), static_cast<uint64_t>(N * C_OUT_PAD)};
    uint64_t output_strides[RANK_OUT - 1] = {
        static_cast<uint64_t>(padded_out_w * sizeof(float)),
        static_cast<uint64_t>(padded_out_w * padded_out_h * sizeof(float))
    };
    uint32_t output_box[RANK_OUT] = {static_cast<uint32_t>(BLOCK_SIZE), static_cast<uint32_t>(BLOCK_SIZE), static_cast<uint32_t>(BLOCK_DEPTH)};
    uint32_t output_elem_stride[RANK_OUT] = {1, 1, 1};

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

__device__ __forceinline__ float* in_tile(float* base, int buf, int y, int x,
        int plane_stride_elems, int pitch_elems)
    {
        return base
            + (size_t)buf * (size_t)plane_stride_elems
            + (size_t)y   * (size_t)pitch_elems
            + (size_t)x;
    }

__global__ void producer_consumer_pattern(
    const __grid_constant__ CUtensorMap input_map,
    const __grid_constant__ CUtensorMap output_map,
    float* kernel,
    float* out,
    int C_OUT_PAD,
    int OUT_H,
    int OUT_W,
    float* inp,
    int H,
    int W,
    int padded_out_w, 
    int stride_y, int stride_x,
    int sh_tile_h, int tile_w_pad, int plane_stride_elems) 
    {

    extern __shared__ __align__(128) unsigned char smem_raw[];
    float* smem_in = reinterpret_cast<float*>(smem_raw);

    __shared__ alignas(128) float smem_out[BLOCK_DEPTH][BLOCK_SIZE][BLOCK_SIZE]; 
    __shared__ float smem_kernel[DEPTH][BLOCK_DEPTH][K][K]; // buffer for BLOCK_DEPTH kernels of corresponding channel
    __shared__ barrier bar_ready[DEPTH]; // track if buffers are ready to be filled,
    __shared__ barrier bar_filled[DEPTH]; // track if buffers are filled-in respectively

    int tid = threadIdx.x;
    const int thread_in_wg = tid % WG_SIZE;
    const int wg = tid / WG_SIZE;

    const bool is_consumer = (wg < CONSUMER_WGS);
    const bool is_producer = (wg >= CONSUMER_WGS) && (wg < CONSUMER_WGS + PRODUCER_WGS);

    // Split blockIdx.z into (n, g_idx)
    constexpr int GROUPS = (OUT_C + BLOCK_DEPTH - 1) / BLOCK_DEPTH;
    const int n = blockIdx.z / GROUPS; // batch index
    const int g_idx = blockIdx.z % GROUPS; // which output channel group
    int oc_offset = g_idx * BLOCK_DEPTH;
    const int g_eff = max(0, min(BLOCK_DEPTH, OUT_C - oc_offset));
    

    static_assert((THREADS_PER_CTA % WG_SIZE) == 0, "CTA must be an integer # of warpgroups");
    static_assert((CONSUMER_WGS + PRODUCER_WGS) * WG_SIZE == THREADS_PER_CTA,
              "Role partition must cover the CTA exactly");

    if (tid < DEPTH) {
        init(bar_ready + tid, THREADS_PER_CTA);
        init(bar_filled + tid, THREADS_PER_CTA);
        cde::fence_proxy_async_shared_cta();
    }

    const int NPIX = BLOCK_SIZE * BLOCK_SIZE;
    // Zero out smem_out
    for (int i = tid; i < g_eff * NPIX; i += THREADS_PER_CTA) {
        int z = i / (NPIX); 
        int inner = i % (NPIX);
        int y = inner / BLOCK_SIZE, x = inner % BLOCK_SIZE;
        smem_out[z][y][x] = 0.0f;
    }
    
    // Sync to make barriers, zeroed out accumulator visible to everyone
    __syncthreads();

    if (is_consumer) {
        // consumer
        // TODO: Add setmaxnreg instruction to increase register count
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(40));
        for (int i = 0; i < DEPTH; ++i) {
            bar_ready[i].arrive(); // buffers are made ready for initial fill
        }
        
        constexpr int SLOTS_MAX = (NPIX + (NUM_CONSUMERS) - 1) / (NUM_CONSUMERS); // how many output pixels each consumer thread calculates, compile time constant
        float acc_slots[SLOTS_MAX][BLOCK_DEPTH]; // thus we can alleviate per thread register pressure by reucing work per block (i.e reducing BLOCK_DEPTH). But that also decreases input reuse
        // zero out accumulators-per-thread
        for (int s = 0; s < SLOTS_MAX; ++s) {
            for (int oc = 0; oc < g_eff; ++oc) {
                acc_slots[s][oc] = 0.0f;
            }
        }

        for (int c = 0; c < IN_C; ++c) {
            int buf = c % DEPTH;
            bar_filled[buf].arrive_and_wait();
            float* tile0 = in_tile(smem_in, buf, 0, 0, plane_stride_elems, tile_w_pad);

            // process all pixels owned by this thread
            for (int i = tid, s = 0; i < NPIX; i += NUM_CONSUMERS, ++s) {
                // calculates top left corner of convolution window
                int ly = i / BLOCK_SIZE;
                int lx = i % BLOCK_SIZE;
                
                for (int ky = 0; ky < K; ++ky) {
                    float* row = tile0 + (ly * stride_y + ky) * tile_w_pad + lx * stride_x;
                    for (int kx = 0; kx < K; ++kx) {
                        float x = row[kx]; // load value to register
                        #pragma unroll
                        for (int oc = 0; oc < g_eff; ++oc) {
                            acc_slots[s][oc] += x * smem_kernel[buf][oc][ky][kx];
                        }
                    }
                }
            }
            barrier::arrival_token token = bar_ready[buf].arrive();
        }

        for (int i = tid, s = 0; i < NPIX; i += NUM_CONSUMERS, ++s) {
            int ly = i / BLOCK_SIZE, lx = i % BLOCK_SIZE;
            for (int oc = 0; oc < g_eff; ++oc) {
                smem_out[oc][ly][lx] = acc_slots[s][oc];
            }
        } 
    }
    else {
        // producer
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32)); // Give up registers with "setmaxnreg"
        for (int c = 0; c < IN_C; ++c) {
            int buf = c % DEPTH; // Determine which buffer to load into
            float* dst = in_tile(smem_in, buf, 0, 0, plane_stride_elems, tile_w_pad);
            bar_ready[buf].arrive_and_wait();
            barrier::arrival_token t_load;
            if (thread_in_wg == 0) {
                // printf("reading channel %d from global for block (%d, %d) \n", c, blockIdx.x, blockIdx.y);
                cde::cp_async_bulk_tensor_3d_global_to_shared(
                    dst,
                    &input_map,
                    blockIdx.x * BLOCK_SIZE * stride_x,
                    blockIdx.y * BLOCK_SIZE * stride_y,
                    n * IN_C + c, // plane = n*IN_C + C
                    bar_filled[buf]
                );
                t_load = cuda::device::barrier_arrive_tx(bar_filled[buf], 1, sh_tile_h * tile_w_pad * sizeof(float));
            }
            else {
                // load corresponding kernels in 
                int rel_tid = tid - NUM_CONSUMERS - 1; // since thread 128 won't execute this part
                for (int i = rel_tid; i < g_eff*K*K; i += (NUM_PRODUCERS - 1)) {
                    // now to determine what it will load and where to. consective threads should load in one whole filter
                    int out_c = i / (K*K);
                    int inner = i % (K*K);
                    int row = inner / K;
                    int col = inner % K;
                    smem_kernel[buf][out_c][row][col] = kernel[(c*OUT_C + (out_c + oc_offset))*K*K + row*K + col];
                }
                t_load = bar_filled[buf].arrive();
            }
        }
    }
    
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // Final TMA store of smem_out to global
    if (tid == 0) {
        cde::cp_async_bulk_tensor_3d_shared_to_global(
            &output_map,
            blockIdx.x*BLOCK_SIZE,
            blockIdx.y*BLOCK_SIZE,
            n * C_OUT_PAD + oc_offset, // store plane = n*C_OUT_PAD + OC_offset
            &smem_out[0][0][0]
        );
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
    }
}

void launch_conv2d_tma_3d(torch::Tensor input, torch::Tensor kernel, torch::Tensor output, int stride_h, int stride_w) {

    int stride_y = stride_h;
    int stride_x = stride_w;
    
    int N = input.size(0);
    int H = input.size(2);
    int W = input.size(3);

    int OUT_H = (H - K) / stride_y + 1;
    int OUT_W = (W - K) / stride_x + 1;

    constexpr int GROUPS = (OUT_C + BLOCK_DEPTH - 1) / BLOCK_DEPTH;
    constexpr int C_OUT_PAD = GROUPS * BLOCK_DEPTH;

    int padded_out_h = ((OUT_H + BLOCK_SIZE - 1)/BLOCK_SIZE)*BLOCK_SIZE;
    int padded_out_w = ((OUT_W + BLOCK_SIZE - 1)/BLOCK_SIZE)*BLOCK_SIZE;

    int sh_tile_h   = (BLOCK_SIZE - 1) * stride_y + K;
    int sh_tile_w   = (BLOCK_SIZE - 1) * stride_x + K;
    int tile_w_pad  = ((sh_tile_w + 3) / 4) * 4;   // 16B row pitch

    size_t plane_elems  = (size_t)sh_tile_h * tile_w_pad;
    size_t plane_stride_elems = ((plane_elems + 31) / 32) * 32;  // align plane to 128B (32 floats)
    size_t smem_bytes   = (size_t)DEPTH * plane_stride_elems * sizeof(float);
    
    // Setup tensor maps
    CUtensorMap input_map;
    CUtensorMap output_map;
    setup_tensor_maps(
        input_map, output_map,
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, IN_C, H, W,
        padded_out_h, padded_out_w,
        C_OUT_PAD,
        sh_tile_h, tile_w_pad,
        BLOCK_SIZE
    );

    // Kernel launch
    dim3 threads(THREADS_PER_CTA);  
    dim3 blocks(
        (OUT_W + BLOCK_SIZE - 1) / BLOCK_SIZE, 
        (OUT_H + BLOCK_SIZE - 1) / BLOCK_SIZE, 
        N * GROUPS  
    );

    producer_consumer_pattern<<<blocks, threads, smem_bytes>>>(
        input_map, output_map,
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        C_OUT_PAD, OUT_H, OUT_W,
        input.data_ptr<float>(),
        H, W,
        padded_out_w,
        stride_y, stride_x,
        sh_tile_h, tile_w_pad, plane_stride_elems
    );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel or TMA error: " << cudaGetErrorString(err) << std::endl;
    }
}
