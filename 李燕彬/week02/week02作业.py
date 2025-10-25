# coding:utf8
"""
课后学习
基于pytorch框架编写训练模型
实现一个自行构造的找规律（机器学习）任务
规律: x是一个多维向量(不能是一维），如果第一个数大于最后个数，则为正样本，反之为负样本

用交叉熵实现一个多分类任务，多维随机向量最大的数字在哪维就属于哪一类
example: x -> [1,2,3,4,5] y -> [0,0,0,0,1]
         x -> [4,3,2,1,0] y -> [1,0,0,0,0]
"""

import torch
import numpy as np
from matplotlib import pyplot as plt

from torch import nn
from transformers.utils import LossKwargs


class TorchModel(nn.Module):
    # def __init__(self, input_size: int = 5, output_size: int = 1):
    def __init__(self, input_size: int = 5, output_size: int = 5):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层, 输入和输出相同维度
        # self.activation = nn.Sigmoid() 归化函数, 主要是二分类
        self.activation = torch.softmax # 多分类使用Softmax，输出与类别数相同维度
        # self.loss = nn.functional.mse_loss  # 损失函数采用均方差损失
        self.loss = nn.CrossEntropyLoss()  # 损失函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, output_size)
        # y_pred = self.activation(x)
        if y is not None:
            return self.loss(x, y)  # 预测值与真实值计算损失
        else:
            # return y_pred  # 输出预测结果
            return self.activation(x, dim=1)  # 推理时输出概率


# 生成一个样本，样本的生成方法，代表了我们要学习的规律
# 随机生成一个多维向量x，和相同维度的y，如果多维向量的x第N维是最大值，那么向量y的第N维值是1，其他是0
def build_sample(size):
    x = np.random.random(size)
    # y = 1 if x[0] > x[-1] else 0  # 二分类 仅有0和1值
    y = [ 1 if i == max(x) else 0 for i in x ] # 多分类，按输入size返回相同维度数据
    return x,y


# 随机生成一批样本
def build_dataset(total_sample_num, input_size):
    X = list()
    Y = list()
    for i in range(total_sample_num):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, input_size):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num, input_size)
    # print(f"本次预测集共有{sum(y)}个正样本，{test_sample_num - sum(y)}个负样本")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip( y_pred, y):  # 与真实标签进行对比
            # 原有二分类判断是否预测正确
            # if float(y_p) < 0.5 and int(y_t) == 0:
            #     correct += 1  # 负样本判断正确
            # elif float(y_p) > 0.5 and int(y_t) == 1:
            #     correct += 1  # 正样本判断正确
            # else:
            #     wrong += 1
            # 现有多分类，判断预测的类别与真实类别是否一致（概率大于0.99认为是该类别）
            y_t_index = list(y_t).index(max(y_t))
            y_p_index = list(y_p).index(max(y_p))
            if y_p_index == y_t_index and (max(y_p.numpy()) - max(y_t.numpy()))**2 < 0.01:
                correct += 1  # 样本判断正确
            else:
                wrong += 1
    print(f"本次预测集共有{test_sample_num}个样本, 正确预测个数: {correct}, 错误个数: {wrong}, 正确率: {correct / test_sample_num}")
    return correct / test_sample_num


# 训练业务逻辑代码
def main(input_size):
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 1000  # 每次训练样本格式
    train_sample = 500000  # 每轮训练总共训练的样本总数
    # input_size = 5  # 输入向量维度，不能小于2
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size=input_size, output_size=input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = list()
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample, input_size)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = list()
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss =model(x,y)    # 计算Loss， model.forward(x,y)
            loss.backward()     # 计算梯度
            optim.step()        # 更新权重
            optim.zero_grad()   # 梯度归零
            watch_loss.append(loss.item())  # 损失值记录
        print(f"============>第{epoch+1}轮平均loss:{np.mean(watch_loss)}")
        acc = evaluate(model, input_size)   # 测试本轮模型接口
        if loss.item() < 0.00001 or acc > 0.99:
            break # 损失率足够低提前结束
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "./model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(input_size, model_path, input_vec):
    # input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    # for vec, res in zip(input_vec, result):
    # 原有二分类结果，不需要做额外处理，直接打印即可
    #     print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果
    #  多分类任务结果，需要预测类别与真实类别对比进行打印
    for vec, y_p in zip(input_vec, result):
        # print(f"vec: {vec}, res: {y_p}")  # 打印结果
        y_t =  [ 1 if i == max(vec) else 0 for i in vec ]
        y_t_index = list(y_t).index(max(y_t)) + 1
        y_p_index = list(y_p).index(max(y_p)) + 1
        correct = y_p_index == y_t_index and (max(y_p.numpy()) - max(y_t)) ** 2 < 0.01
        msg = "正确" if correct else "错误"
        print(f"输入:{vec}, 预测类别:第{y_p_index}维, 实际类别:第{y_t_index}维, 预测结果{msg}, 概率值: {max(y_p.numpy())*100}")


if __name__ == "__main__":
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894],
                [0.42735869,0.83904471,0.21465618,0.06062571,0.06084349],
                [0.30885331,0.85786346,0.54208813,0.90038646,0.72640395],
                [0.26881238,0.60675519,0.85280065,0.57306828,0.04737965],
                [0.81574041,0.54862784,0.13178178,0.79022274,0.42345878],
                [0.74287733,0.87459213,0.69047244,0.35854304,0.10841768],
                [0.41386441,0.22431992,0.09216637,0.16391365,0.03990295],
                [0.21592015,0.23645238,0.10190512,0.74825176,0.06536708]
    ]
    set_size = len(test_vec[0])
    main(set_size)
    predict(set_size, "model.bin", test_vec)

