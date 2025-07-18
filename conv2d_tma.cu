#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

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
                       int OUT_H, int OUT_W,
                       int BLOCK_SIZE) {
    constexpr uint32_t RANK = 3;

    uint64_t input_dims[RANK] = {static_cast<uint64_t>(IN_C), static_cast<uint64_t>(H), static_cast<uint64_t>(W)};
    uint64_t input_strides[RANK - 1] = {
        static_cast<uint64_t>(H * W * sizeof(float)),
        static_cast<uint64_t>(W * sizeof(float))
    };
    uint32_t input_box[RANK] = {1, static_cast<uint32_t>(SH_TILE_W), static_cast<uint32_t>(SH_TILE_W)};
    uint32_t input_elem_stride[RANK] = {1, 1, 1};

    uint64_t output_dims[RANK] = {1, static_cast<uint64_t>(OUT_H), static_cast<uint64_t>(OUT_W)};
    uint64_t output_strides[RANK - 1] = {
        static_cast<uint64_t>(OUT_H * OUT_W * sizeof(float)),
        static_cast<uint64_t>(OUT_W * sizeof(float))
    };
    uint32_t output_box[RANK] = {1, static_cast<uint32_t>(BLOCK_SIZE), static_cast<uint32_t>(BLOCK_SIZE)};
    uint32_t output_elem_stride[RANK] = {1, 1, 1};

    auto encode = get_cuTensorMapEncodeTiled();

    CUresult res = encode(
        &input_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT,
        RANK,
        d_input,
        input_dims,
        input_strides,
        input_box,
        input_elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    assert(res == CUDA_SUCCESS);

    res = encode(
        &output_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT,
        RANK,
        d_output,
        output_dims,
        output_strides,
        output_box,
        output_elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    assert(res == CUDA_SUCCESS);
}


__device__ void producer(const __grid_constant__ CUtensorMap tensor_map, barrier ready[], barrier filled[], float* smem_buffer, int H, int W, int buffer_len)
{
    for (int c = 0; c < IN_C; ++c) {
        ready[c%2].arrive_and_wait(); /* wait for buffer_(c%2) to be ready to be filled */
        /* produce, i.e., fill in, buffer_(c%2)  */
        cde::cp_async_bulk_tensor_3d_global_to_shared(&smem_buffer[c % 2], &tensor_map, c, blockIdx.y * SH_TILE_W, blockIdx.x * SH_TILE_W, filled[c%2]);
        barrier::arrival_token token = cuda::device::barrier_arrive_tx(filled[c%2], 1, buffer_len * sizeof(float)); /* buffer_(c%2) is filled */
    }
}

__device__ void consumer(barrier ready[], barrier filled[], float* smem_buffer, float* smem_out, float* smem_kernel, int buffer_len)
{
    barrier::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
    barrier::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid == 0) return; // it is the producer thread
    
    int num_threads = blockDim.x * blockDim.y;
    int out_pixels = BLOCK_SIZE * BLOCK_SIZE;
    for (int i = tid - 1; i < out_pixels; i += num_threads - 1) {
        int out_y = i / BLOCK_SIZE;
        int out_x = i % BLOCK_SIZE;

        // Init local accumulator
        for (int c = 0; c < IN_C; ++c) {
            // float acc = 0.0f;
            int buf_idx = c % 2;
            barrier::arrival_token token = filled[buf_idx].arrive()
            filled[buf_idx].wait(std::move(token));
            /* consume buffer_(c%2) */
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float val = smem_buffer[buf_idx][0][out_y + ky][out_x + kx];
                    float weight = smem_kernel[c][ky][kx];
                    smem_out[0][out_y][out_x] += val * weight;
                }
            }
            token = ready[buf_idx].arrive(); /* buffer_(c%2) is ready to be re-filled */
            
        }
    }   
}

//N is the total number of float elements in arrays in and out
__global__ void producer_consumer_pattern(int N, int buffer_len, float* in, float* out) {

    // Shared memory buffer declared below is of size 2 * buffer_len
    // so that we can alternatively work between two buffers.
    // buffer_0 = buffer and buffer_1 = buffer + buffer_len
    // __shared__ extern float buffer[];
    __shared__ alignas(128) float smem_buffer[2][1][SH_TILE_W][SH_TILE_W];

    __shared__ alignas(128) float smem_out[1][BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float smem_kernel[IN_C][K][K];

    // bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
    // while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively
    __shared__ barrier bar[4];
    
    auto block = cooperative_groups::this_thread_block();
    if (block.thread_rank() < 2){
            init(bar + block.thread_rank(), block.size());
    } else if (block.thread_rank() < 4){
        init(bar + block.thread_rank(), 1);
    }  
    // block.sync();

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_weights = IN_C * K * K;
    for (int i = tid; i < total_weights; i += blockDim.x * blockDim.y) {
        int c = i / (K * K);
        int k = i % (K * K);
        int ky = k / K;
        int kx = k % K;
        smem_kernel[c][ky][kx] = kernel[c * K * K + ky * K + kx];
    }

    __syncthreads();
    
    if (tid == 0)
        producer(bar, bar+2, buffer, in, N, buffer_len);
    else
        consumer(bar, bar+2, buffer, out, N, buffer_len);
}