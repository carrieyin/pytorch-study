import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear

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

print(mouleTest)
