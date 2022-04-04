import torchvision
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0]
print(img.shape)
print(target)

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        step += 1
