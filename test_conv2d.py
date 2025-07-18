import torch
import torch.nn.functional as F
import my_cuda_conv  # This is your compiled CUDA extension

# Image and kernel sizes
# H, W = 32, 32       # Input image size
# K = 3               # Kernel size (assumed square)

# # Create input image and kernel
# input = torch.randn(H, W, dtype=torch.float32, device='cuda').contiguous()
# kernel = torch.randn(K, K, dtype=torch.float32, device='cuda').contiguous()

H = W = 128
K = 3

input = torch.arange(H * W, dtype=torch.float32, device='cuda').reshape(H, W)
kernel = torch.ones(K, K, dtype=torch.float32, device='cuda')


# Output size: valid convolution
out_H = H - K + 1
out_W = W - K + 1
output = torch.empty(out_H, out_W, dtype=torch.float32, device='cuda')

# Call your CUDA kernel
# my_cuda_conv.conv2d_naive(input, kernel, output)
my_cuda_conv.conv2d_shared(input, kernel, output)


# Compute expected result using PyTorch for comparison
# Need to unsqueeze to get (N=1, C=1, H, W) shape for F.conv2d
input_pt = input.unsqueeze(0).unsqueeze(0)     # Shape: (1, 1, H, W)
kernel_pt = kernel.unsqueeze(0).unsqueeze(0)   # Shape: (1, 1, K, K)
output_ref = F.conv2d(input_pt, kernel_pt).squeeze()  # Remove batch/channel dims

# Compare
max_error = (output - output_ref).abs().max().item()
print(f"Max absolute error: {max_error:.6f}")

# Optional: print values for debugging
print("CUDA output:\n", output)
print("Reference output:\n", output_ref)
print("Difference: \n", (output - output_ref))
