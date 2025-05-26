import torch

#数据生成函数
def Generate_Data(w, b, num):
    """
    生成数据 y = wx + b

    @:param
        W:权重。
        b:偏置。
        num:生成数量。

    @:return
         特征与标签。
    """

    X = torch.normal(0, 1, (num, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

#模型参数
true_w = torch.tensor([6, -7.6, 36, 2, -3.4])
true_b = 4.2

#生成数据
features, labels = Generate_Data(true_w, true_b, 1000)

#保存数据
torch.save([features,labels], "./linear_data.pt")
