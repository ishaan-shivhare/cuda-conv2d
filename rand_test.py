import torch
import torch.nn.functional as F

x = torch.randn(1, 4, 128, 128, device='cuda')
w = torch.randn(1, 4, 3, 3, device='cuda')

out = F.conv2d(x, w)
