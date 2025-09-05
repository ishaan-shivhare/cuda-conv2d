import torch
import torch.nn.functional as F
import my_cuda_conv  # Your compiled CUDA extension

torch.backends.cudnn.enabled = False    # cuDNN optimizations lead to small error/difference in output

# Configuration
N = 4
IN_C = 32
OUT_C = 128
H = W = 512
K = 3
BLOCK_SIZE = 16
BLOCK_DEPTH = 16
sy = 1
sx = 1

# Create multi-channel input and kernel
input = torch.arange(N * IN_C * H * W, dtype=torch.float32, device='cuda').reshape(N, IN_C, H, W)
kernel = torch.randn(OUT_C, IN_C, K, K, dtype=torch.float32, device='cuda')  # Use ones for easy reference
kernel_oihw = kernel 
kernel_iohw = kernel_oihw.permute(1, 0, 2, 3).contiguous()

# print(input.shape)

# Output shape
out_H = (H - K) // sy + 1
out_W = (W - K) // sx + 1
# output = torch.empty(out_H, out_W, dtype=torch.float32, device='cuda')
padded_h = ((out_H + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
padded_w = ((out_W + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
C_OUT_PAD = ((OUT_C + BLOCK_DEPTH - 1) // BLOCK_DEPTH) * BLOCK_DEPTH
output = torch.zeros(N, C_OUT_PAD, padded_h, padded_w, device='cuda', dtype=torch.float32)  # padded for TMA alignment
# Call the CUDA kernel
# my_cuda_conv.conv2d_shared_multi_in(input, kernel, output)
# my_cuda_conv.conv2d_tma(input, kernel, output, padded_h, padded_w)
my_cuda_conv.conv2d_tma_3d(input, kernel_iohw, output, sy, sx)

# Reference with PyTorch

print("input.shape:", input.shape)     
print("kernel.shape:", kernel.shape)   

output_ref = F.conv2d(input, kernel, stride=(sy, sx)) 


# Compare only the valid region
cuda_valid = output[:, :OUT_C, :out_H, :out_W]
ref_valid = output_ref  

# Compare
max_error = (cuda_valid - ref_valid).abs().max().item()
print(f"Max absolute error: {max_error:.6f}")

# Optional debugging
print("CUDA output:\n", cuda_valid)
print("Reference output:\n", output_ref)
print("Difference:\n", cuda_valid - output_ref)

# ---------------- timing: your kernel ----------------
# warmup, iters = 5, 30
# for _ in range(warmup):
#     output.zero_()
#     my_cuda_conv.conv2d_tma_3d(input, kernel, output, sy, sx)
# torch.cuda.synchronize()

# start = torch.cuda.Event(True); end = torch.cuda.Event(True)
# start.record()
# for _ in range(iters):
#     output.zero_()
#     my_cuda_conv.conv2d_tma_3d(input, kernel, output, sy, sx)
# end.record()
# torch.cuda.synchronize()
# ms_custom = start.elapsed_time(end) / iters
# print(f"[Custom kernel] {ms_custom:.3f} ms")

# # ---------------- timing: cuDNN FP32 ----------------
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.allow_tf32 = False  # keep it FP32

# for _ in range(warmup):
#     F.conv2d(input, kernel, stride=(sy, sx))
# torch.cuda.synchronize()

# start = torch.cuda.Event(True); end = torch.cuda.Event(True)
# start.record()
# for _ in range(iters):
#     F.conv2d(input, kernel, stride=(sy, sx))
# end.record()
# torch.cuda.synchronize()
# ms_cudnn = start.elapsed_time(end) / iters
# print(f"[cuDNN FP32]    {ms_cudnn:.3f} ms")