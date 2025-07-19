import torch
import torch.nn.functional as F
import my_cuda_conv  # Your compiled CUDA extension

# Configuration
IN_C = 3
H = W = 128
K = 3

# Create multi-channel input and kernel
input = torch.arange(IN_C * H * W, dtype=torch.float32, device='cuda').reshape(IN_C, H, W)
kernel = torch.ones(IN_C, K, K, dtype=torch.float32, device='cuda')  # Use ones for easy reference

# Output shape
out_H = H - K + 1
out_W = W - K + 1
output = torch.empty(1, out_H, out_W, dtype=torch.float32, device='cuda') # added 1 for the TMA version. Naive is without channel dim

# Call the CUDA kernel
# my_cuda_conv.conv2d_shared_multi_in(input, kernel, output)
my_cuda_conv.conv2d_tma(input, kernel, output)

# Reference with PyTorch
input_pt = input.unsqueeze(0)      # (1, C, H, W)
kernel_pt = kernel                 # (C, K, K)
kernel_pt = kernel_pt.unsqueeze(0)  # (1, C, K, K)
output_ref = F.conv2d(input_pt, kernel_pt) # for TMA we keep the batch dim # .squeeze()  # remove batch dim
output_ref = output_ref.squeeze(1)  # (1, H_out, W_out) to match CUDA TMA output
# Compare
max_error = (output - output_ref).abs().max().item()
print(f"Max absolute error: {max_error:.6f}")

# Optional debugging
print("CUDA output:\n", output)
print("Reference output:\n", output_ref)
print("Difference:\n", output - output_ref)

