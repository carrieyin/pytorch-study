# by y_dd
# 时间 2023/07/06
import torch as torch
import torchvision.datasets
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
import torch.nn as nn
import datetime

from torch.utils.tensorboard import SummaryWriter

now = datetime.datetime.now()
dataset = torchvision.datasets.CIFAR10("../../../dataset/cifar-10", train=True, transform=torchvision.transforms.ToTensor(), download=True)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=2)

print(dataset[0][0].shape)
print(dataset[0][0])

class ClassMoudle(nn.Module):
    def __init__(self):
        super(ClassMoudle, self).__init__()
        self.module1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10))

    def forward(self, x):
        x= self.module1(x)
        return x

loss = nn.CrossEntropyLoss()
mouleTest=ClassMoudle()

optim = torch.optim.SGD(mouleTest.parameters(), lr=0.01)

index=0
writer = SummaryWriter("logs")
for epoch in range(10):
    print("train: epoch:{}".format(epoch))
    for data in dataloader:
        image, target = data
        #print(image)
        print(image.shape)
        output = mouleTest(image)
        lossresult = loss(output, target)
        optim.zero_grad()
        lossresult.backward()
        optim.step()
        # print("index:{}, loss:{}".format(index, lossresult))
        writer.add_scalar("loss", lossresult.item(), index)
        index = index +1
    epoch = epoch + 1

current = datetime.datetime.now()
print("traning data used:{}".format(current - now))

torch.save(mouleTest.state_dict(), "../../target/moudle_save/class_10_1.pth")
