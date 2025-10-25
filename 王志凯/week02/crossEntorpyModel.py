import torch
import torch.nn as nn
import numpy as np

"""
规则：假设一组五维向量，如果第一个数最大，则为第1类样本......第五个数最大则为第5类样本
要求：1.使用交叉熵进行模型训练，找出规律；2.使用训练好的模型对样本进行测试
"""

# 构造训练数据（为了方便后面获取测试数据，当need_Y=0时，只构造X的数据）
def build_datas(data_size, need_Y=1):
    X = torch.rand(data_size,5)
    if need_Y == 0:
        return X
    # 构造Y的数据
    Y = torch.zeros_like(X)
    for x, y in zip(X, Y):
        max_x_index = torch.argmax(x)
        y[max_x_index] = 1
    return X, Y

# 定义模型和参数（线性层，激活函数-softmax，loss函数-交叉熵）
class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        # 输出是类似[1,0,0,0,0]类型，也是五维向量
        self.Linear = nn.Linear(input_size, 5)
        # torch的交叉熵函数已经内置了softmax的计算，不需要torch.Softmax额外定义激活函数
        self.loss = nn.CrossEntropyLoss()

    # 模型实际调用的方法，该方法是进行loss计算的真实方法
    def forward(self, x, y=None):
        linear = self.Linear(x)
        if y is not None:
            # 如果传入y值，则认为是在进行模型训练，返回loss对象
            return self.loss(linear, y)
        else:
            # 如果没有传入y值，则进行预测（torch的交叉熵内置softmax，而预测的时候不需要进行交叉熵的计算，所以使用torch.softmax单独计算）
            pred_y = torch.softmax(linear, dim=1)
            return pred_y

# 对每轮训练好的模型进行准确率检测
def monitor(model):
    model.eval()
    correct = 0
    wrong = 0
    X = build_datas(100, need_Y=0)
    Y = model.forward(X)
    for x, y in zip(X, Y):
        max_index_x = torch.argmax(x)
        max_index_y = torch.argmax(y)
        if max_index_x == max_index_y:
            correct += 1
        else:
            wrong += 1
    print(f"该轮预测总样本数：{correct+wrong}，正确率：{correct/(correct + wrong)} \n ============")

# 使用训练好的模型对数据进行预测
def predict(X):
    model = MyModel(5)
    model.load_state_dict(torch.load("model.pt"))
    # 不计算梯度
    model.eval()
    with torch.no_grad():
        correct = 0
        wrong = 0
        Y_pred = model.forward(X)
        for x, y in zip(X, Y_pred):
            print(f"原数据：{x}，预测数据：{y} ", end="")
            max_index_x = torch.argmax(x)
            max_index_y = torch.argmax(y)
            if max_index_x == max_index_y:
                correct += 1
                print("预测正确")
            else:
                wrong += 1
                print("预测错误")
        print(f"总样本数：{correct+wrong}，正确率：{correct/(correct + wrong)}")

# 模型训练
def main():
    total_round = 20       # 总训练轮数
    data_size = 5000       # 总样本数
    batch_size = 20        # 一批样本的个数（多少个样本进行一次权重更新）
    learning_rate = 0.01    # 学习率
    input_size = 5         # 输入数据的维度（单组数据的长度）

    # 初始化模型
    model = MyModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 构造训练数据
    X, Y = build_datas(data_size)

    for i in range(total_round):
        # 开启训练模式，model.eval()测试模式
        model.train()
        # 记录loss值，用于打印每轮loss变化
        loss_round = []
        # 总数据批次
        batch_total = X.shape[0] // batch_size
        # 多余的样本
        batch_remain = X.shape[0] % batch_size
        # 对每批次数据进行训练
        for batch in range(batch_total + 1):
            if batch != batch_total:
                batch_X = X[batch * batch_size : (batch + 1) * batch_size]
                batch_Y = Y[batch * batch_size : (batch + 1) * batch_size]
            else:
                # 剩余的样本
                batch_X = X[-batch_remain :]
                batch_Y = Y[-batch_remain :]
            # 将样本传入模型获取loss对象
            batch_loss = model(batch_X, batch_Y)
            # 根据loss计算梯度
            batch_loss.backward()
            # 使用优化器更新权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
            loss_round.append(batch_loss.item())
        print(f"训练第{i+1}轮，loss平均值：{np.mean(loss_round)}")
        if np.mean(loss_round) <= 0.1:
            break
        # 对每轮训练好的模型进行准确率检测
        monitor(model)
    # 保存最终模型
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    # 设置全局打印参数
    torch.set_printoptions(
        # 对float类型的保留位数
        precision=5,
        # 不启用科学技术
        sci_mode=False
    )
    # main()

    # 检测模型训练结果
    X = build_datas(1000, need_Y=0)
    predict(X)
