# softmax回归

softmax回归是解决分类问题的模型。那它是如何解决分类问题的呢？就是通过计算输出概率的最大可能性来判断这个样本是属于哪一个类别的，每个输出会对应每个特征有不同的权重进行计算然后进行分类，Ssoftmax输出层是一个全连接层。

<img src="https://mmbiz.qpic.cn/sz_mmbiz_png/7TWRhh4xickk4AGn7eGvE35AFsAmXZzJ9Mxw1F14f9tr7iakic3RVkSh3dibo0Isngo8yFJ0Jm3JltklQlAbL2Sz0w/640?wx_fmt=png&from=appmsg" style="zoom:50%;" />
$$
softmax(X)_{ij}=\frac{exp(X_{ij})}{\sum_{k}{esp(X_{ik})}}
$$
$$

全连接层输出后通过softmax运算后得到概率，以此概率确定类别。

## 一、损失函数

### 1、交叉熵

交叉熵采用真实标签的预测概率的负对数似然
$$
P(X|Y) = \prod_{i=1}^{n}{P(Y^{(i)}|X^{(i)})}
$$

$$
L(y,\hat{y})=-\sum_{i=0}^{n}y_i\log(\hat{y_i})
$$

在训练softmax回归模型后，给出任何样本特征，我们可以预测每个输出类别的概率。 通常我们使用预测概率最高的类别作为输出类别。 如果预测与实际类别（标签）一致，则预测是正确的。 在接下来的实验中，我们将使用*精度*（accuracy）来评估模型的性能。 精度等于正确预测数与预测总数之间的比率

## 二、训练数据集获取

### 1.Fashion-MNIST数据集

我们将使用Fashion-MNIST数据集 ([Xiao *et al.*, 2017](https://zh.d2l.ai/chapter_references/zreferences.html#id189))进行模型训练。Fashion-MNIST由10个类别的图像组成， 每个类别由*训练数据集*（train dataset）中的6000张图像 和*测试数据集*（test dataset）中的1000张图像组成。 因此，训练集和测试集分别包含60000和10000张图像。 测试数据集不会用于训练，只用于评估模型性能。Fashion-MNIST中包含的10个类别，分别为t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。

### 2、获取代码封装

从Fashion-MNIST数据集中获取训练集与测试集，然后分批次读取训数据与测试数据，每个批次32张灰度图，28*28像素点，为了方便训练时使用，封装函数。

```python
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import mxnet as d2l

def load_data_fashion_mnist(batch_size, resize=None):
    """
    下载Fashion-MNIST数据集，然后将其加载到内存中

    batch_size:批次大小
    resize:改变数据形状resize*resize
    return:返回训练集与测试集
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=4))


```

d2l中已经封装了读取Fashion-MNIST数据集中的方法

```python
#数据获取
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```



## 三、softmax回归简洁实现

```python
import torch
from torch import nn
from d2l import torch as d2l
#数据读取
batch_size = 32
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#模型定义
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
#初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
#损失函数
loss = nn.CrossEntropyLoss(reduction='none')
#优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
#训练
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```



















