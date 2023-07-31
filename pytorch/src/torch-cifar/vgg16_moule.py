# by y_dd
# 时间 2023/07/10
import torch as torch
import torchvision.models

vgg16_true = torchvision.models.vgg16()
print('ok')
print(vgg16_true)

torch.save(vgg16_true, "../../../../moudule/vgg16_moule_1.pth")

torch_dict = torch.save(vgg16_true.state_dict(), "../../../../moudule/vgg16_moule_2.pth")

vgg16 = vgg16_true.load_state_dict(torch.load("../../../../moudule/vgg16_moule_2.pth"))
print(vgg16)