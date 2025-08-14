#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include "conv2d_config.h"


// Naive Conv 2d for grayscale images - no padding single output channel
__global__ void conv2d_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* output,
    int H, int W, int K
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (idx < (W - K + 1) && idy < (H - K + 1)){
        float accum = 0.0f;
        for (int y_offset = 0; y_offset < K; y_offset++){
            for(int x_offset = 0; x_offset < K; x_offset++){
                int col = idx + x_offset;
                int row = idy + y_offset;
                accum += input[row * W + col] * kernel[y_offset * K + x_offset];
            }
        }
        output[idy * (W - K + 1) + idx] = accum;
    }
}

// #define BLOCK_SIZE 16
// #define K 3
// #define SH_TILE_W (BLOCK_SIZE + K - 1)

__global__ void conv2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* output,
    int H, int W
) {
    // 1. Declare shared memory tile
    __shared__ float sh_input[SH_TILE_W][SH_TILE_W];
    // 2. Compute base_row / base_col
    int base_row = blockIdx.y * BLOCK_SIZE;
    int base_col = blockIdx.x * BLOCK_SIZE;
    // 3. Cooperative tile load from global → shared
    for (int i = threadIdx.y; i < SH_TILE_W; i += BLOCK_SIZE){
        for (int j = threadIdx.x; j < SH_TILE_W; j += BLOCK_SIZE){
            if ((base_row + i) < H && (base_col + j) < W)
                sh_input[i][j] = input[(base_row + i) * W + (base_col + j)];
            else
                sh_input[i][j] = 0.0f; // Zero-pad the halo if it spills over
        }
    }
    // 4. Load kernel into registers
    float reg_kernel[K * K];
    for (int i = 0; i < 9; i++) {
        reg_kernel[i] = kernel[i];
    }
    // 5. __syncthreads()
    __syncthreads();
    // 6. Compute 1 output pixel per thread from shared tile
    int idy = base_row + threadIdx.y;
    int idx = base_col + threadIdx.x;
    if (idx < (W - K + 1) && idy < (H - K + 1)){
        float accum = 0.0f;
        for (int y_offset = 0; y_offset < K; y_offset++){
            for(int x_offset = 0; x_offset < K; x_offset++){
                int col = threadIdx.x + x_offset;
                int row = threadIdx.y + y_offset;
                accum += sh_input[row][col] * reg_kernel[y_offset * K + x_offset];
            }
        }
        // 7. Write result to output
        output[idy * (W - K + 1) + idx] = accum;
    }

}

__global__ void conv2d_shared_multi_in_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* output,
    int H, int W
) {
    // 1. Declare shared memory tile
    __shared__ float sh_input[SH_TILE_W][SH_TILE_W][IN_C];
    // 2. Compute base_row / base_col
    int base_row = blockIdx.y * BLOCK_SIZE;
    int base_col = blockIdx.x * BLOCK_SIZE;
    // 3. Cooperative tile load from global → shared
    for (int c = threadIdx.z; c < IN_C; c += blockDim.z){
        for (int i = threadIdx.y; i < SH_TILE_W; i += blockDim.y){
            for (int j = threadIdx.x; j < SH_TILE_W; j += blockDim.x){
                if ((base_row + i) < H && (base_col + j) < W)
                    sh_input[i][j][c] = input[(c * H * W) + (base_row + i) * W + (base_col + j)];
                else
                    sh_input[i][j][c] = 0.0f; // Zero-pad the halo if it spills over
            }
        }
    }
    // __syncthreads();
    
    __shared__ float sh_output[BLOCK_SIZE * BLOCK_SIZE];

    // Initialising the output block to zero
    if (threadIdx.z == 0) {
    sh_output[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float reg_kernel[K * K];
    int idy = base_row + threadIdx.y;
    int idx = base_col + threadIdx.x;
    for (int c = threadIdx.z; c < IN_C; c += blockDim.z){
        // 4. Load kernel into registers
        for (int i = 0; i < K * K; i++) {
            reg_kernel[i] = kernel[(c * K * K) + i];
        }
        // 5. Compute the convolution for each channel
        if (idx < (W - K + 1) && idy < (H - K + 1)){
            float accum = 0.0f;
            for (int y_offset = 0; y_offset < K; y_offset++){
                for(int x_offset = 0; x_offset < K; x_offset++){
                    int col = threadIdx.x + x_offset;
                    int row = threadIdx.y + y_offset;
                    accum += sh_input[row][col][c] * reg_kernel[y_offset * K + x_offset];
                }
            }
            // sh_output[threadIdx.y * blockDim.x + threadIdx.x] += accum;
            // should replace with atomicAdd_block scope instead
            atomicAdd(&sh_output[threadIdx.y * blockDim.x + threadIdx.x], accum);   // if we guarantee IN_C <= 32, we could assign x axis to channel dim and do a warp shuffle reduction
        }
    }
    __syncthreads();
    
    // 6. Write to Global Memory
    if (threadIdx.z == 0 && idx < (W - K + 1) && idy < (H - K + 1)){
        output[idy * (W - K + 1) + idx] = sh_output[threadIdx.y * blockDim.x + threadIdx.x];
    }
}

// Host launcher
void launch_conv2d_naive(torch::Tensor input, torch::Tensor kernel, torch::Tensor output) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(kernel.device().is_cuda(), "kernel must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "kernel must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");

    int H = input.size(0);
    int W = input.size(1);
    int K = kernel.size(0); // assuming square

    const int out_H = H - K + 1;
    const int out_W = W - K + 1;

    dim3 blockDim(16, 16);
    dim3 gridDim((out_W + 15) / 16, (out_H + 15) / 16);

    conv2d_naive_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        H, W, K
    );

    // Optional: kernel launch error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
}

void launch_conv2d_shared(torch::Tensor input, torch::Tensor kernel, torch::Tensor output) {
    // 1. Compute output dimensions
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(kernel.device().is_cuda(), "kernel must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "kernel must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    

    int H = input.size(0);
    int W = input.size(1);
    // int K = kernel.size(0); // assuming square

    const int out_H = H - K + 1;
    const int out_W = W - K + 1;
    TORCH_CHECK(output.size(0) == out_H && output.size(1) == out_W, "output tensor has incorrect shape");


    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (out_W + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (out_H + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    // 2. Launch conv2d_shared_kernel with appropriate grid/block
    conv2d_shared_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        H, W
    );

    // Optional: kernel launch error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
}

void launch_conv2d_shared_multi_in(torch::Tensor input, torch::Tensor kernel, torch::Tensor output) {
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

    int H = input.size(1);
    int W = input.size(2);
    int K = kernel.size(1);

    int out_H = H - K + 1;
    int out_W = W - K + 1;

    TORCH_CHECK(output.size(0) == out_H && output.size(1) == out_W, "output tensor has incorrect shape");

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, K);
    dim3 gridDim(
        (out_W + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (out_H + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    conv2d_shared_multi_in_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        H, W
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
}
