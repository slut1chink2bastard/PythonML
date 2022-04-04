import torch
import torch.nn.functional as F
# TODO:figure out the meaning of the convolution
input = torch.tensor([[1, 2, 3, 4, 1], [0, 2, 3, 4, 0], [3, 5, 6, 433, 3], [423, 3, 3, 2, 1], [5, 3, 1, 13, 3]])

kernel = torch.tensor([[1, 2, 3], [9, 0, 2], [2, 4, 4]])
# print(input.shape)
# print(kernel.shape)
res_input = torch.reshape(input, (1, 1, 5, 5))
res_kernel = torch.reshape(kernel, (1, 1, 3, 3))

output = F.conv2d(res_input, res_kernel, stride=1)
print(output)
