import torch
from torch import nn
from torch.utils import data


#读取数据
features, labels = torch.load("./linear_data.pt")

input = features.shape[1]
output = labels.shape[1]

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)


#定义模型
net = nn.Sequential(nn.Linear(input, output))

#初始化参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

#损失函数
loss = nn.MSELoss()

#优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')