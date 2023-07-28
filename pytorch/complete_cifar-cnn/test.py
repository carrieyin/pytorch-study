import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.models
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter

test_dataset = torchvision.datasets.CIFAR10("../../../dataset/cifar-10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(test_dataset, batch_size=64)

mouleTest = Sequential(
    Conv2d(3, 32, 5, padding=2),
    MaxPool2d(2),
    Conv2d(32, 32, 5, padding=2),
    MaxPool2d(2),
    Conv2d(32, 64, 5, padding=2),
    MaxPool2d(2),
    Flatten(),
    Linear(1024, 64),
    Linear(64, 10))

if not os.path.exists("../moudle_save/class_10_1.pth"):
    print("error exist")
    exit(-1)

mouleTest.load_state_dict(torch.load("../moudle_save/class_10_1.pth"), strict=False)

writer = SummaryWriter("./logs")
step = 0
with torch.no_grad():
    for data in test_dataloader:
        image, target = data
        output = mouleTest(image)
        accureNum = (output.argmax(1) == target).sum()
        print(output.shape)
        print(output)
        print(output.argmax(1))
        break
        print(target)
        print(accureNum)
        break
        writer.add_scalar("acuraccy", accureNum / image.size(0), step)
        step = step + 1
