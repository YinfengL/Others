from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter("logs")
image_path = "/Users/mac/Desktop/学习/土堆 pytorch入门/Learn_Pytorch/dataset/hymenoptera_data/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np. array(img_PIL)
print(type(img_array))
print(img_array.shape)
writer.add_image("test", img_array, 1, dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
