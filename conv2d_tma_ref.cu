#include <cuda/barrier>
#include <cooperative_groups.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ void producer(barrier ready[], barrier filled[], float* buffer, float* in, int N, int buffer_len)
{
    for (int i = 0; i < (N/buffer_len); ++i) {
        ready[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */
        /* produce, i.e., fill in, buffer_(i%2)  */
        barrier::arrival_token token = filled[i%2].arrive(); /* buffer_(i%2) is filled */
    }
}

__device__ void consumer(barrier ready[], barrier filled[], float* buffer, float* out, int N, int buffer_len)
{
    barrier::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
    barrier::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */
    for (int i = 0; i < (N/buffer_len); ++i) {
        filled[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */
        /* consume buffer_(i%2) */
        barrier::arrival_token token = ready[i%2].arrive(); /* buffer_(i%2) is ready to be re-filled */
    }
}

//N is the total number of float elements in arrays in and out
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
    block.sync();

    // For now: 1 producer warpgroup and 1 consumer warpgroup (16x16 block)
    // Can increase block size to 16x32 for 3 consumer 1 producer
    if (tid < warpSize * 4) {
        // consumer
        barrier::arrival_token token1 = bar_ready[0].arrive(); // buffer 0 is ready for initial fill 
        barrier::arrival_token token2 = bar_ready[1].arrive(); // buffer 1 is ready for initial fill 
        for (int c = 0; c < IN_C; ++c) {
            bar_filled[c%2].arrive_and_wait();
            float (*cur_buf)[TILE_W_PAD] = (c%2==0) ? smem_buf0 : smem_buf1;
            for (int global_y = blockIdx.y * BLOCK_SIZE + threadIdx.y; global_y < OUT_H; global_y += BLOCK_SIZE) {
                for (int global_x = blockIdx.x * BLOCK_SIZE + threadIdx.x; global_x < OUT_W; global_x += BLOCK_SIZE) {
                    float accum = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            int sy = global_y - blockIdx.y * BLOCK_SIZE + ky;
                            int sx = global_x - blockIdx.x * BLOCK_SIZE + kx;
                            if (sy < SH_TILE_W && sx < SH_TILE_W) {
                                accum += cur_buf[sy][sx] * smem_kernel[c][ky][kx];
                            }
                        }
                    }
                    smem_out[global_y - blockIdx.y * BLOCK_SIZE][global_x - blockIdx.x * BLOCK_SIZE] += accum;
                }
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