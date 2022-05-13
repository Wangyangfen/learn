# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:00:04 2022

@author: Administrator
"""

import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
#训练1轮
EPOCH =1                # train the training data n times, to save time, we just train 1 epoch
#一次前向传播训练里面用64个样本
BATCH_SIZE = 64
TIME_STEP = 28          # 表示序列本身的长度 在mnist中 图片对应的TIME_STEP是28
INPUT_SIZE = 28         # 输入的维度
LR = 0.01               # 学习率
DOWNLOAD_MNIST = True   # 如果没有下载数据集则会进行下载操作

loss_list=[]#保存损失
acc_list=[]#保存准确率
iteration_list=[] #循环次数


# Mnist digital dataset
train_data = dsets.MNIST(
    root='./data',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # 格式类型的转换，将numpy类型转换为tensor类型Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)


print(train_data.train_data.size())     # (60000, 28, 28) 60000张28X28的图片
print(train_data.train_labels.size())   # (60000) 60000个标签
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')#cmap='gray'图片颜色为灰白
plt.title('%i' % train_data.train_labels[0])
#plt.show()


#创建数据集的可迭代对象（一个batch一个batch 的读取数据，相当于一批一批（64）的读取数据）
#torch.utils.data.DataLoader数据读取的一个重要接口  Shuffle : 是否打乱数据位置，当为Ture时打乱数据
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = dsets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
#取其中2000个样本进行归一化处理作为测试集
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255. # shape (2000, 28, 28) value in range(0,1)
#取出2000个标签并转化为numpy类型（标签和样本一一对应）作为测试标签
test_y = test_data.test_labels.numpy()[:2000]   


#定义RNN模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__() #初始化父类的构造方法
        #对变量进行赋值
        self.rnn = nn.LSTM(         
            input_size=INPUT_SIZE, #定义输入的维度
            hidden_size=64,         # 隐层的维度
            num_layers=1,           # 1层RNN
            batch_first=True,       #RNN输入的固定格式batch_size在中间，设置为true可以将batch_size调到第一位 (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10) #全连接层（隐层和输出）
    #前向传播
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        #分离隐藏状态，避免梯度爆炸
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        #输出，-1表示只输出最后的隐藏层的一个状态
        out = self.out(r_out[:, -1, :])
        return out

#创建模型对象
rnn = RNN()
print(rnn)
#构造一个优化器对象Optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   #（模型参数和学习率）
#定义损失函数
loss_func = nn.CrossEntropyLoss()                       

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):     #b_x, b_y这一批数据的图片和标签
        #将数据转换成RNN的输入维度
        b_x = b_x.view(-1, 28, 28)   #数据的转换 重新定义矩阵的形状reshape x to (batch, time_step每批的跨度28, input_size输入的维度28)
        #输入数据进行前向传播 得出预测结果output
        output = rnn(b_x)                               
        #计算损失(output预测结果，b_y真实值)
        loss = loss_func(output, b_y)                   
        optimizer.zero_grad()                           # 清空之前的梯度（否则会不断累加）
        #进行反向传播求梯度
        loss.backward()                                 
        # 根据梯度更新参数
        optimizer.step()                                




        #模型的验证(每循环50次在测试集上做一次模型的验证)
        if step % 50 == 0:
            #利用测试集得出预测结果test_output
            test_output = rnn(test_x)
            #获取预测中概率值最大的值的下标                   
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            #计算模型的准确度  .astype()方法在不同的数值类型之间相互转换
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            #每训练一次，将损失值添加到数组 便于之后的绘图
            loss1 = loss.data.numpy()#将数值从tensor类型转换成numpy类型
            loss_list.append(loss1)
            #将迭代次数添加到数组
            iteration_list.append(step)
            #将准确率保存到数组 便于绘图
            acc_list.append(accuracy)
            
            epoch=epoch+1
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

#迭代次数和Loss的图
#定义画布
fig=plt.figure()
#用plot()函数绘制迭代次数(x轴)和Loss(y轴)的关系图
plt.plot(iteration_list,loss_list)
#x轴标签
plt.xlabel('Number of Iteration')
#y轴标签
plt.ylabel('Loss')
#标题
plt.title('RNN')
plt.show()
#迭代次数和Acc的图
#定义画布
fig1=plt.figure()
plt.plot(iteration_list,acc_list)
plt.xlabel('Number of Iteration')
plt.ylabel('Acc')
plt.title('RNN')
plt.show()
