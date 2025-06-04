import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "/Users/mac/Desktop/学习/土堆 pytorch入门/Learn_Pytorch/imgs/dog.jpg"
image = Image.open(image_path)
print(image)
#image = image.convert("RGB")

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                             torchvision.transforms.ToTensor()])

image = transforms(image)
print(image.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.model(x)
        return x
#转化回cpu
model = torch.load("tudui_30_gpu.pth", map_location = torch.device('cpu'))
print(model)
image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))