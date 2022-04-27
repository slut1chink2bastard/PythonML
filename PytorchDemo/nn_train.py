import torch
import torchvision
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from model import NeuralNetwork

# prepare data
train_data = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(train_data_size)
print(test_data_size)

# load the data
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# neural network
network = NeuralNetwork()

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = SGD(network.parameters(), lr=0.01)

# setting parameters
epoch = 10
for i in range(10):
    print("starting the {}".format(i) + " round")
    for data in train_dataloader:
        img, target = data
        output = network(img)
        loss = loss_fn(output, target)
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("step {}".format(i) + ",training loss {}".format(loss))
        i += 1
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            output = network(img)
            test_loss = loss_fn(output, targets)
    print("test loss {}".format(test_loss))


