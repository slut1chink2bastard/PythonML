import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5], [-1, 3]])
output = torch.reshape(input, (-1, 1, 2, 2))
print(output.shape)
print(input)

class NNDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()
    def forward(self,input):
        return self.relu(input)


nn_demo = NNDemo()
output = nn_demo(input)
print(output)


