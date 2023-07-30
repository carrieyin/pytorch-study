# by y_dd
# 时间 2023/07/06
import torch as torch
import torchvision.datasets
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

WRITER_ADD_SCALAR = writer.add_scalar("acuraccy", accureNum / image.size(0), step)

dataset = torchvision.datasets.CIFAR10("../../../dataset/cifar-10", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10("../../../dataset/cifar-10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
print(len(dataset))
print(len(test_dataset))
dataloader = DataLoader(dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

print(dataset[0][0].shape)

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

optim = torch.optim.SGD(mouleTest.parameters(),lr=0.01)

index=0
writer = SummaryWriter("loss_1")
for epoch in range(20):
    for data in dataloader:
        image, target = data
        #print(target)
        output = mouleTest(image)
        lossresult = loss(output, target)
        optim.zero_grad()
        lossresult.backward()
        optim.step()
        # print("index:{}, loss:{}".format(index, lossresult))
        writer.add_scalar("loss", lossresult.item(), index)
        index = index +1

train_loss = 0
step = 0
torch.save(ClassMoudle, "./moule_save/class_10.pth")
accureNum = 0
with torch.no_grad():

    for data in test_dataloader:
        image, target = data
        output = mouleTest(image)
        loss_result = loss(output, target)
        writer.add_scalar("train_loss", loss_result.item(), step)
        accureNum = (output.argmax(1) == target).sum()

        writer.add_scalar("train_loss", loss_result.item(), step)
        step = step + 1
writer.close()