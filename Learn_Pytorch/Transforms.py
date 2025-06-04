from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transfo rms
img_path = "/Users/mac/Desktop/学习/土堆 pytorch入门/Learn_Pytorch/dataset/data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)
writer.close()