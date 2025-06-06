import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *

train_data = torchvision.datasets.CIFAR10(root="../data", train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False,transform=torchvision.transforms.ToTensor(),
                                          download=True)

train_data_size = len(train_data)
print("训练数据集的长度为:{}".format(train_data_size))
test_data_size = len(test_data)
print(f"监测数据集的长度为{test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64,sha)


tudui = Tudui()


loss_fn = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("../logs_train")


for i in range(epoch):
    print("------第{}论训练开始------".format(i+1))
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1

        if total_train_step % 100 == 0:
            print("训练次数：{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(),total_train_step)


     tudui.eval()
    total_test_loss = 0
    total_accuracy = 0


    with torch.no_grad():
        for data in test_dataloader:
         imgs, targets = data
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












