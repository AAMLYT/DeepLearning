import torch
import random

#生成数据
def generate_data(w,b,num_examples):
    """
    按照w与b生成数据
    :param w: 权重值
    :param b: 偏置值
    :param num_examples: 生成样本数
    :return:返回特征与标签
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([3, 1.6, -66])
true_b = 3.2
features, labels = generate_data(true_w, true_b, 1000)

#读取数据
batch_size = 10
def data_iter(batch_size, features, labels):
    """
    分批次读取数据
    :param batch_size:一个批次多少数据
    :param features:传入特征
    :param labels:传入标签
    :return:返回含有特征与标签的小批量数
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # 打乱读取顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

#初始化模型参数
pred_w = torch.normal(0,1,size=(3,1), requires_grad=True)
pred_b = torch.zeros(1, requires_grad=True)

#定义模型
def linreg(X, w, b):
    """
    线性回归模型
    :param X: 特征
    :param w: 权重
    :param b: 偏置
    :return: X*w+b
    """
    return torch.matmul(X, w) + b

#定义损失函数
def squared_loss(y_hat, y):
    """
    均方损失
    :param y_hat: 预测标签
    :param y: 真实标签
    :return: 损失
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

#定义优化算法
def sgd(params, lr, batch_size):
    """
    小批量梯度下降
    :param params: 模型参数
    :param lr: 学习率
    :param batch_size: 批次数量大小
    :return: n
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
#训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, pred_w, pred_b), y)  # X和y的小批量损失
        l.sum().backward()
        sgd([pred_w, pred_b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, pred_w, pred_b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

""""
练习：
1、如果我们将权重初始化为零，会发生什么。算法仍然有效吗？
    权重为零，这个模型是一个单层线性模型，在进行参数优化时与初始化参数无关，所以算法有效
    多层神经网络中，在更新权值的过程中，代价函数对不同权值参数的偏导数相同 ，所以算法无法收敛，就无效了

2、为什么在squared_loss函数中需要使用reshape函数？
    y真实值的维度与y预测值维度不相同，需要转换形状
    
3、如果样本个数不能被批量大小整除，data_iter函数的行为会有什么变化？
    最后一个批次的数量会有所改变
"""