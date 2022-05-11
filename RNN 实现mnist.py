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
EPOCH =1                # train the training data n times, to save time, we just train 1 epoch
#一次前向传播训练里面用64个样本
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data

loss_list=[]#保存损失
acc_list=[]#保存准确率
iteration_list=[] #循环次数


# Mnist digital dataset
train_data = dsets.MNIST(
    root='./data',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data Loader for easy mini-batch return in training
#torch.utils.data.DataLoader数据读取的一个重要接口  Shuffle : 是否打乱数据位置，当为Ture时打乱数据
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = dsets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
#归一化处理
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255. # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]    # covert to numpy array


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit隐藏节点的个数
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10) #全连接层

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)
#构造一个优化器对象Optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, 28, 28)              # 重新定义矩阵的形状reshape x to (batch, time_step, input_size)

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # 清空之前的梯度
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # 根据梯度更新参数





        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()#返回test_output中所有元素的最大值，并转化为numpy
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)

            loss1 = loss.data.numpy()
            loss_list.append(loss1)

            iteration_list.append(step)
            acc_list.append(accuracy)
            epoch=epoch + 1
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

#迭代次数和Loss的图
fig1=plt.figure()
plt.plot(iteration_list,loss_list)
plt.xlabel('Number of Iteration')
plt.ylabel('Loss')
plt.title('RNN')
plt.show()
#迭代次数和Acc的图
fig2=plt.figure()
plt.plot(iteration_list,acc_list)
plt.xlabel('Number of Iteration')
plt.ylabel('Acc')
plt.title('RNN')
plt.show()
