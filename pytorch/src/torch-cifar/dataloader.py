# by y_dd
# 时间 2023/06/26
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.CIFAR10(root="../../../dataset/cifar-10", train=True,download=True, transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("c100")

step=0
for data in data_loader:
    imgs, tag = data
    #print(img)
    #print(img.size())

    writer.add_images("c100_image", imgs, step)
    step = step + 1

writer.close()
#print(image)
#image.show()