import torch
import time
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#定义训练设备
device = torch.device("cuda")


train_data = torchvision.datasets.CIFAR10(root="../data", train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False,transform=torchvision.transforms.ToTensor(),
                                          download=True)

train_data_size = len(train_data)
print("训练数据集的长度为:{}".format(train_data_size))
test_data_size = len(test_data)
print(f"监测数据集的长度为{test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


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


tudui = Tudui()
tudui =tudui.to(device)
if torch.cuda.is_available():
   tudui.cuda()

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()


learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)


total_train_step = 0
total_test_step = 0
epoch = 10


writer = SummaryWriter("../logs_train")

start_time = time.time()
for i in range(epoch):
    print("------第{}论训练开始------".format(i+1))
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1

        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(),total_train_step)


    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0


    with torch.no_grad():
        for data in test_dataloader:
         imgs, targets = data

         imgs = imgs.to(device)
         targets = targets.to(device)
         outputs = tudui(imgs)
         loss = loss_fn(outputs, targets)
         total_test_loss += loss.item()
         accuracy = (outputs.argmax(1) == targets).sum()
         total_accuracy += accuracy


    print(f'整体测试集上的Loss:{total_test_loss}')
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1


    torch.save(tudui,"tudui_{}.pth".format(i+1))
#   torcj.save(tudui.statte_dict(),"tudui_{}.pth".format(i+1))
    print("模型已保存")



writer.close()












