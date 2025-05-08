# LeNet卷积神经网络

## 一、LeNet组成

### 1、LeNet由两个部分组成：

​	卷积编码器：由两个卷积层组成;

​	全连接层密集块：由三个全连接层组成。

![](D:\code\网络模型\LeNet.png)

## 二、实现

```python
import torch
from torch import nn
from d2l import torch as d2l

#获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

#模型定义
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

#模型精度评估
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    使用GPU计算模型在数据集上的精度

    @param
        net:模型
        data_iter:数据
        device:使用cpu还是gpu评估模型精度

    @return
        accuracy:精度
    """
    #判断net是否为Module类或者子类
    if isinstance(net, nn.Module):
        # 设置为评估模式
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    # 记录正确预测的数量与总预测的数量实例化
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # 记录正确预测的数量与总预测的数量
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

#定义训练函数
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """
    用GPU训练模型(在第六章定义)

    @param
        net:模型
        train_iter:训练数据
        test_iter:测试数据
        num_epochs:迭代周期个数
        lr:学习率
        device:使用什么设备进行训练

    @return

    """
    #初始化参数
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    #设置在什么设备上训练
    print('training on', device)
    net.to(device)
    #优化算法
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    #损失函数
    loss = nn.CrossEntropyLoss()
    #可视化
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    #训练耗时与训练数据大小
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 记录训练损失之和，训练准确率之和，样本数实例化
        metric = d2l.Accumulator(3)
        #进入训练模式
        net.train()
        for i, (X, y) in enumerate(train_iter):
            #开始计时
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            #停止计时
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            #绘制训练损失与精度
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        #测试集在模型上的精度
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        #绘制测试精度
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'训练损失 {train_l:.3f}, 训练精度 {train_acc:.3f}, 'f'测试精度 {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')
    d2l.plt.show()
#训练模型
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
