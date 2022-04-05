from torch import nn
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataLoader = DataLoader(dataset, batch_size=64)


class NNDemo(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self,x):
        return self.conv1(x)


nn_demo = NNDemo()
print(nn_demo)

for data in dataLoader:
    imgs, tragets = data
    output = nn_demo(imgs)
    print(imgs.shape)

    print(output.shape)
