import torch
import torchvision
from torch import nn
from torch.nn import L1Loss
from torch.utils.data import DataLoader

from PytorchDemo.nn_seq import ConNNSeq

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataLoader = DataLoader(dataset, batch_size=64)
# test normal loss
input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss = L1Loss(reduction="mean")
result = loss(input, target)
print(result)

mse_loss = nn.MSELoss()
result_mse = mse_loss(input, target)
print(result_mse)

# real nn loss
loss = nn.CrossEntropyLoss()
nn_seq = ConNNSeq()
for data in dataLoader:
    img, target = data
    output = nn_seq(img)
    result_loss = loss(output, target)
    result_loss.backward()
    print(result_loss)
