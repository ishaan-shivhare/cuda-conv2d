import torch
import torch.nn.functional as F
import my_cuda_conv  # Your compiled CUDA extension

# Configuration
IN_C = 4
H = W = 128
K = 3
BLOCK_SIZE = 16

# Create multi-channel input and kernel
input = torch.arange(IN_C * H * W, dtype=torch.float32, device='cuda').reshape(IN_C, H, W)
kernel = torch.ones(IN_C, K, K, dtype=torch.float32, device='cuda')  # Use ones for easy reference

# Output shape
out_H = H - K + 1
out_W = W - K + 1
# output = torch.empty(out_H, out_W, dtype=torch.float32, device='cuda')
padded_h = ((out_H + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
padded_w = ((out_W + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
output = torch.zeros(padded_h, padded_w, device='cuda', dtype=torch.float32)  # padded for TMA alignment
# Call the CUDA kernel
# my_cuda_conv.conv2d_shared_multi_in(input, kernel, output)
my_cuda_conv.conv2d_tma(input, kernel, output, padded_h, padded_w)

# Reference with PyTorch
input_pt = input.unsqueeze(0)      # (1, C, H, W)
kernel_pt = kernel                 # (C, K, K)
kernel_pt = kernel_pt.unsqueeze(0)  # (1, C, K, K)

print("input_pt.shape:", input_pt.shape)     # should be (1, 4, 128, 128)
print("kernel_pt.shape:", kernel_pt.shape)   # should be (1, 4, 3, 3)

output_ref = F.conv2d(input_pt, kernel_pt).squeeze()  # remove batch dim

# Compare only the valid region
cuda_valid = output[:out_H, :out_W]
ref_valid = output_ref.squeeze()  # shape: (out_H, out_W)

# Compare
max_error = (cuda_valid - ref_valid).abs().max().item()
print(f"Max absolute error: {max_error:.6f}")

# Optional debugging
print("CUDA output:\n", output)
print("Reference output:\n", output_ref)
print("Difference:\n", output - output_ref)

