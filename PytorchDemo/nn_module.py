from torch import nn
import torch


class nnDemo(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return input + 1


nnTest = nnDemo()
x = torch.tensor(1)
output = nnTest(x)
print(output)

