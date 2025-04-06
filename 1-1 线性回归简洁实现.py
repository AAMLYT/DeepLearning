import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

#按照给出的真实权重与偏置生成1000个数据
true_w = torch.tensor([3, 1.6, -66])
true_b = 3.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
#生成样本
def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个PyTorch数据迭代器
    data_arrays:数据输入
    batch_size:每个批次数据返回多少个样本数据
    is_train:是否打乱数据
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
#模型选择
net = nn.Sequential(nn.Linear(3,1))
#参数初始化
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
#损失函数
loss = nn.MSELoss(reduction='mean')#reduction 平均损失
#loss = nn.MSELoss(reduction='sum')#reduction 总损失
#优化算法：小批量梯度下降，学习率lr：0.03
learning_rate = 0.03
trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)
'''
# 动态调整学习率
adjusted_learning_rate = learning_rate / batch_size
trainer.param_groups[0]['lr'] = adjusted_learning_rate
# 查看学习率
print(f"Adjusted Learning Rate: {adjusted_learning_rate}")
'''
#模型训练,3次
num_epochs = 3
for epoch in range(num_epochs):
    g = 0
    for X, y in data_iter:
        l = loss(net(X) ,y)#前向传播
        trainer.zero_grad()#梯度值归0
        l.backward()#反向传播
        trainer.step()#更新梯度值
        #print(net[0].weight.grad)
        #print(net[0].bias.grad)
    print(f'epoch {epoch + 1}, loss {l:f}')
#查看训练值与真实值
w = net[0].weight.data
b = net[0].bias.data
print('真实w：', true_w)
print('真实b：', true_b)
print('预测w：', w.reshape(true_w.shape))
print('预测b：', b)

"""
练习：
1、如果将小批量的总损失替换为小批量损失的平均值，需要如何更改学习率？
小批量损失的平均值比小批量总损失的平均值小，所以要把学习率降小，可以除以批量的大小
#定义优化器，假设初始学习率为0.03
loss = nn.MSELoss(reduction='sum')#reduction 小批量总损失
learning_rate = 0.03
trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)
#动态调整学习率
adjusted_learning_rate = learning_rate / batch_size
trainer.param_groups[0]['lr'] = adjusted_learning_rate
#查看学习率
print(f"Adjusted Learning Rate: {adjusted_learning_rate}")
num_epochs = 3
for epoch in range(num_epochs):
    g = 0
    for X, y in data_iter:
        l = loss(net(X) ,y)#前向传播
        trainer.zero_grad()#梯度值归0
        l.backward()#反向传播
        trainer.step()#更新梯度值
    print(f'epoch {epoch + 1}, loss {l:f}')
    
2、查看深度学习框架文档，它们提供了哪些损失函数和初始化方法？用Huber损失代替原损失。
损失函数：
l1_loss = nn.L1Loss()：L1 Loss计算预测值与真实值之间的绝对差值。适用于对异常值鲁棒性要求较高的场景。
cross_entropy_loss = nn.CrossEntropyLoss()：交叉熵损失（Cross-Entropy Loss）常用于分类任务，特别是多分类问题。它结合了Softmax和负对数似然损失。
bce_loss = nn.BCEWithLogitsLoss()：BCEWithLogitsLoss将Sigmoid层和二分类交叉熵损失合并在一起，提高了数值稳定性。适用于二分类任务。
kl_loss = nn.KLDivLoss()：KL散度损失（KL Divergence Loss）用于衡量两个概率分布之间的差异。常用于生成模型和概率模型。
smooth_l1_loss = nn.SmoothL1Loss()：平滑L1损失（Smooth L1 Loss）结合了L1和L2损失的优点，适用于回归任务，特别是目标检测中的位置回归。
focal_loss = FocalLoss()：Focal Loss用于处理类别不平衡问题，特别适用于目标检测任务。它通过调整每个样本的权重，使模型更关注难分类的样本。

权重初始化：
Xavier 初始化：Xavier初始化方法适用于Sigmoid和Tanh激活函数，保持每层输出方差与输入方差相同。
Kaiming 初始化：Kaiming初始化方法适用于ReLU及其变种激活函数，保持数据尺度在适当范围。
均匀分布初始化：从给定范围内的连续均匀分布中采样值。
正态分布初始化：从具有指定均值和标准差的正态分布中采样值。

Huber损失代替原损失：
loss = nn.SmoothL1Loss(beta=0.5)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        #开始计算梯度
        trainer.zero_grad()
        l.backward()
        trainer.step()  #对所有参数进行更新
    print("epoch: {}, loss:{}".format(epoch + 1, l))

3、如何访问线性回归的梯度？
net[0].weight.grad
net[0].bias.grad
"""
