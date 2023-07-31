# by y_dd
# 时间 2023/06/26
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.CIFAR10(root="../../../dataset/cifar-10", train=True,download=True)
#print(train_set[0])
#data_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("c100")
to_tensor = transforms.ToTensor()
for i in range(10):
    img, tag = train_set[i]
    image_tensor = to_tensor(img)
    #print(image_tensor.shape)
    writer.add_image("c100_image", image_tensor, i)

writer.close()
#print(image)
#image.show()