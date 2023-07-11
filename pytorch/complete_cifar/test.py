import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models

test_dataset = torchvision.datasets.CIFAR10("../../../dataset/cifar-10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(test_dataset, batch_size=64)

mouleTest = torch.load_state_dict(torch.load("../moule_save/class_10_1.pth"))

print(mouleTest)
