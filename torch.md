# pytorch

## 一、常用张量生成

### 1、torch.tensor

```python
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)
"""
把data转换为张量格式

@Param:
    data：传入数据，可以是列表、元组、矩阵
    dtype：返回张量数据类型
    device：返回张量属于哪个设备（cpu、cuda）
	requires_grad：是否自动求导
	pin_memory： 张量分配内存位置
@return
	tensor
"""
Data = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]], requires_grad=False)
tensor([[ 0.1000,  1.2000],
        [ 2.2000,  3.1000],
        [ 4.9000,  5.2000]])
```

### 2、torch.zeros

```python
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
"""
创建一个填充值全为0的张量

@Param:
    size：一个定义输出张量形状的整数序列。可以是可变数量的参数，也可以是列表或元组等集合。
    out：输出张量
    layout：张量的布局
    dtype：返回张量数据类型
    device：返回张量属于哪个设备（cpu、cuda）
	requires_grad：是否自动求导
@return
	tensor
"""
Data = torch.zeros(2, 3)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
```

### 3、torch.normal

```python
torch.normal(mean, std, size, *, out=None) 
"""
返回一个张量，其中的随机数从各自的正态分布中抽取，这些分布的均值和标准差已给定。

@Param:
    mean：均值
    std：标准差
    size：定义输出张量形状的整数序列
@return
	tensor
"""
Data = torch.normal(2, 3, size=(1, 4))
tensor([[-1.3987, -1.9544,  3.6048,  0.7909]])
```

### 4、torch.rand

```python
torch.rand(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
"""
返回一个填充了在区间 ([0,1) 上均匀分布的随机数的张量

@Param:
	size:定义输出张量形状的整数序列
    generator：伪随机数生成器
    out：输出张量
    dtype：返回张量数据类型
    layout:张量的布局
    device:返回张量属于哪个设备（cpu、cuda）
    requires_grad:是否自动求导
@return
	tensor
"""
Data = torch.rand(2, 3)
tensor([[ 0.8237,  0.5781,  0.6879],
        [ 0.3816,  0.7249,  0.0998]])
```

### 5、torch.randn

```python
torch.randn(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
"""
返回一个张量，其元素是从均值为 0、方差为 1 的正态分布中随机采样得到的。

@Param:
	size:定义输出张量形状的整数序列
    generator：伪随机数生成器
    out：输出张量
    dtype：返回张量数据类型
    layout:张量的布局
    device:返回张量属于哪个设备（cpu、cuda）
    requires_grad:是否自动求导
@return
	tensor
"""
Data = torch.randn(2, 3)
tensor([[ 1.5954,  2.8929, -1.0923],
        [ 1.1719, -0.4709, -0.1996]])
```

## 二、数据保存与读取

### 1、torch.save

```python
torch.save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)
"""
将对象保存到磁盘文件。

@Param:
	obj:要保存的对象
    f：保存文件路径
    pickle_module：
    dtpickle_protocolype：
    _use_new_zipfile_serialization：
@Note：
	PyTorch 的一个常用约定是使用 .pt 文件扩展名保存张量
	
"""
x = torch.tensor([0, 1, 2, 3, 4])
torch.save(x, "tensor.pt")
```

### 2、torch.load

```python
torch.load(f, map_location=None, pickle_module=pickle, *, weights_only=True, mmap=None, **pickle_load_args)
"""
从文件加载使用 torch.save() 保存的对象

@Param:
	f：读取文件路径
    map_location：指定如何重新映射存储位置
    pickle_module：
    weights_only：
    mmap：
@Note：
	默认情况下，我们将字节字符串解码为 utf-8。
@return
	Any
"""
torch.load("tensors.pt",
           map_location=lambda storage, 
           loc: storage.cuda(1),
           weights_only=True)
```

三、





# torch.nn

## 一、容器

### 1、Class torch.nn.Module

所有神经网络模块的父类

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

### 2、Sequential

一个 Sequential 容器。模块将按照它们在构造函数中传入的顺序被添加到其中。

```python
class torch.nn.Sequential(*args: Module)
class torch.nn.Sequential(arg: OrderedDict[str, Module])

model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
```

### 3、ModuleList

在一个列表中保存子模块，可以像常规 Python 列表一样进行索引。

1. append(*module*)
   - 将给定模块添加到列表末尾。
2. extend(*modules*)
   - 将 Python 可迭代对象中的模块添加到列表末尾。
3. insert(*index*, *module*)
   - 在列表中给定索引之前插入给定模块。

```python
class torch.nn.ModuleList(modules=None)
```

### 4、ModuleDict

将子模块保存在一个字典中，可以像常规 Python 字典一样进行索引。

1. clear()
   * 从 ModuleDict 中删除所有项。
2. items()
   - 返回 ModuleDict 键值对的可迭代对象。
3. keys()
   - 返回 ModuleDict 键的可迭代对象。
4. pop(*key*)
   - 从 ModuleDict 中移除键并返回其模块。
5. update(*modules*)
   - 使用来自映射的键值对更新 ，覆盖现有键。
6. values()
   - 返回 ModuleDict 值的可迭代对象。

```python
class torch.nn.ModuleDict(modules=None)
```

### 5、ParameterList

以列表形式持有参数, 可以像常规的 Python 列表一样使用。

1. append(*value*)
   - 在列表末尾添加给定值。
2. extend(*values*)
   - 将 Python 可迭代对象中的值添加到列表末尾。

```python
class torch.nn.ParameterList(values=None)
```

### 6、ParameterDict

..........

```python
class torch.nn.ParameterDict(parameters=None)
```

## 二、卷积

### 1、Conv1d

- 一维卷积主要用于处理序列数据，如时间序列和自然语言处理。它在输入序列上滑动一个一维滤波器，计算每个位置的加权和

  ```python
  class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
  ```

  

### 2、Conv2d

- 二维卷积主要用于处理图像数据。它在输入图像上滑动一个二维滤波器，计算每个位置的加权和。二维卷积可以处理单通道和多通道图像。对于多通道图像，每个滤波器在每个通道上分别进行卷积，然后将结果相加，形成一个输出通道。

  ```python
  class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
  ```

  

### 3、Conv3d

- 三维卷积主要用于处理视频数据或三维医学图像。它在输入数据的深度、高度和宽度三个方向上滑动一个三维滤波器，计算每个位置的加权和。三维卷积可以捕捉数据的时序性和空间关联性。

  ```python
  class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
  ```

.........................................................

## 三、池化

### 1、MaxPool1d & MaxPool2d & MaxPool3d

用于降低特征图的维度。它通过定义一个空间邻域（通常为矩形区域），并对该邻域内的特征进行统计处理最大值，从而生成新的特征图。池化操作紧随卷积层之后，能够降低计算量和参数数量，提高计算效率。

### 2、AvgPool1d & AvgPool2d & AvgPool3d

用于降低特征图的维度。它通过定义一个空间邻域（通常为矩形区域），并对该邻域内的特征进行统计处理平均值，从而生成新的特征图。池化操作紧随卷积层之后，能够降低计算量和参数数量，提高计算效率。
