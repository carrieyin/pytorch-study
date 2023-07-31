# by y_dd
# 时间 2023/06/04
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

image_path = "../../resources/image/dog.png"
image = Image.open(image_path)
#image.show()

# numpy 更改图片类型为numpy array
image_array = np.array(image)
print(image_array.shape)
writer = SummaryWriter("logs")
writer.add_image("test", image_array, 1, dataformats='HWC')
writer.add_image("test", image_array, 2, dataformats='HWC')
#writer.add_image()

# tranaforms 更改图片类型为tensor类型
totensor = transforms.ToTensor()
image_tensor = totensor(image)
print(image_tensor)
writer.add_image("test", image_tensor, 3)


# class Student:
#     def __call__(self, name):
#         print("call__init__"+ name)
#
#     def test(self, name): 
#         print("test {}", name);
#
# student = Student()
# student("test")

#trans_compose = transforms.Compose([transforms.RandomCrop(512)], transforms.ToTensor())

writer.close()