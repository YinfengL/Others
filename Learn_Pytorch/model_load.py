import torch
import torchvision
from torch import nn
from model_save import *


#加载模型1

# model = torch.load("vgg16_method1.pth")
# print(model)




#加载模型2
# vgg16 = torchvision.models.vgg16(weights=None)
# vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

#陷阱1


model = torch.load("tudui_method1.pth",weights_only=False)
print(model)
