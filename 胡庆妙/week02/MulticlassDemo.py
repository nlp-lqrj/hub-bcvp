# coding:utf8

# 解决 OpenMP 库冲突问题
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
基于pytorch框架编写模型训练，实现一个找规律(机器学习)的任务。
规律：x是一个5维向量，如果第i个数最大，那么x为i类样本（0<=i<=4）。 比如，x中的第2个数最大，则x为2类样本. (你事先不知道有这个规律，但希望通过AI找到规律。) 
"""


class TorchModel(nn.Module):
    def __init__(self, dim_num):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(dim_num, dim_num)  # 线性层, 输入是5维向量, 输出也是5维
        # self.activation = torch.softmax  # softmax适应用于多分类问题，但由于下面交叉熵loss函数已经包含了softmax逻辑，所以这里不需要再使用softmax
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # shape: [batch_size, dim_num] -> [batch_size, dim_num]
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失, 注意：y是真实标签, shape为[batch_size], 这样才符合交叉熵Loss的传参要求
        else:
            return y_pred  # 输出预测结果


def build_sample(dim_num):
    """
    随机生成一个样本。规律：如果第i个数最大，那么x为i类样本(0<=i<=n-1)。

    Parameters:
    n (int): 向量的维度

    Returns:
    tuple: (x, label) 其中x是n维向量，label标签类别(0<=label<=n-1)
    """
    x = np.random.random(dim_num)
    return x, int(np.argmax(x))  # 直接返回最大值的索引


def build_dataset(total_sample_num, dim_num):
    """生成训练数据集，即生成一批样本"""
    xx = []  # shape: [total_sample_num, dim_num]
    yy = []  # shape: [total_sample_num]     # 注意：这里的y是真实标签, 它不是one-hot的, 这样才符合交叉熵Loss函数的传参要求
    for i in range(total_sample_num):
        x, y = build_sample(dim_num)
        xx.append(x)
        yy.append(y)
    return torch.FloatTensor(np.array(xx)), torch.LongTensor(np.array(yy))


def evaluate(model):
    """测试、评估模型的准确率"""
    model.eval()  # 设置模型为评估模式，等同于 model.train(False)
    test_sample_num = 100
    xx, yy = build_dataset(test_sample_num, 5)

    stat = np.zeros(5)
    for label in yy:
        stat[label] += 1
    print(
        f"本次预测集有{int(stat[0])}个0类样本，{int(stat[1])}个1类样本，{int(stat[2])}个2类样本，{int(stat[3])}个3类样本，{int(stat[4])}个4类样本")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(xx)  # 模型预测 model.forward(x)
        for y_pred, y_true in zip(y_pred, yy):
            pred_class = torch.argmax(y_pred)
            if pred_class == y_true:
                correct += 1
            else:
                wrong += 1
        print(f"正确预测个数：{correct}，正确率：{correct * 100 / (correct + wrong)}%")
        return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    dim_num = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(dim_num)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample, dim_num)
    # 训练过程
    print("\n训练开始...")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20]  train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]  # shape: [batch_size, dim_num]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]  # shape: [batch_size]
            # print("x", x)
            # print("y", y)
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度清零
            watch_loss.append(loss.item())  # 从损失张量中提取出具体的数值

        print(f"=========\n第{epoch + 1}轮的平均loss: {round(np.mean(watch_loss), 5)}")
        avg_loss = np.mean(watch_loss)  # 本轮的平均Loss
        accuracy = evaluate(model)  # 测试模型的准确率
        log.append([accuracy, float(avg_loss)])

    # 保存模型
    torch.save(model.state_dict(), "multiclass_model.bin")

    # 画图
    print("\n每轮训练的准确率及损失:\n", log)

    print("\n绘制准确率及损失曲线...")
    plt.plot(range(len(log)), [item[0] for item in log], label="accuracy")  # 画accuracy曲线
    plt.plot(range(len(log)), [item[1] for item in log], label="loss")  # 画loss曲线
    plt.legend(title='深度学习-多分类模型训练')

    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统下中文可以使用'SimHei'或'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.show()
    return


def predict(model_path, input_samples, dim_num=5):
    """使用训练好的模型做预测 """
    model = TorchModel(dim_num)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_samples))  # 模型预测. result.shape: [batch_size, dim_num]

    for vec, res in zip(input_samples, result):
        print(f"输入：{vec}, 输出:{res.numpy().round(5)}, 预测类别：{torch.argmax(res)}")


if __name__ == "__main__":
    main()
    print("\n使用训练好的模型做预测...")
    some_samples = [[0.078886, 0.152275, 0.310123, 0.035017, 0.889843],
                    [0.749633, 0.552425, 0.956507, 0.957507, 0.848681],
                    [0.007864, 0.672528, 0.835847, 0.345372, 0.191392],
                    [0.993776, 0.594169, 0.925291, 0.467412, 0.188943],
                    [0.907864, 1.672528, 0.135847, 0.345372, 0.191392],
                    [1.993776, 0.594169, 0.925291, 0.467412, 0.188943]
                    ]
    predict("multiclass_model.bin", some_samples)
