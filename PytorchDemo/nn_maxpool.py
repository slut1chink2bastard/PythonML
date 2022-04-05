import torch
from torch import nn
from torch.nn import MaxPool2d

# intuition
input = torch.tensor([[1, 2, 0, 3, 1], [0, 1, 2, 3, 1], [1, 2, 1, 0, 0], [5, 2, 3, 1, 1], [2, 1, 0, 1, 1]],dtype=torch.float32)
input = torch.reshape(input,(-1,1,5,5))
print(input.shape)

class NNDemo(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size=3,ceil_mode=True)
    def forward(self,input):
        return self.maxpool(input)


nn_demo = NNDemo()
output = nn_demo(input)

