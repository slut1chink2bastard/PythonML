import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

'''
Structure of CIFAR10-quick model
'''


class ConNN(nn.Module):
    def __init__(self):
        super().__init__()
        # divide and conquer
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        return self.linear2(self.linear1(
            self.flatten(self.maxpool3(self.conv3(self.maxpool2(self.conv2(self.maxpool1(self.conv1(x)))))))))


con_nn_1 = ConNN()
print(con_nn_1)

# validation
input = torch.ones(64, 3, 32, 32)
output = con_nn_1(input)
print(output.shape)


class ConNNSeq(nn.Module):
    def __init__(self):
        super().__init__()
        # divide and conquer
        self.model = Sequential(Conv2d(3, 32, 5, padding=2), MaxPool2d(2), Conv2d(32, 32, 5, padding=2), MaxPool2d(2),
                                Conv2d(32, 64, 5, padding=2), MaxPool2d(2), Flatten(), Linear(1024, 64), Linear(64, 10))

    def forward(self, x):
        return self.model(x)


con_nn_2 = ConNNSeq()
print(con_nn_2)

# validation
input = torch.ones(64, 3, 32, 32)
output = con_nn_2(input)
print(output.shape)
