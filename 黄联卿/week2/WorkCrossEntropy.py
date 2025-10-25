# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本
本周作业:改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        # self.activation = torch.sigmoid  # 使用交叉熵时，不能用sigmoid（sigmoid适用于二分类或多标签分类）。
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            # print("标签形状：", y.shape)
            # print("forward预测值{}，真实值{}",y_pred,y)
            # print("self.loss", self.loss(x, y))
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return x  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大的数字在哪维就属于哪一类
def build_sample():
    x = np.random.random(5)
    # 找到最大值所在的索引（维度）
    max_index = np.argmax(x)
    # 返回向量和对应的类别（索引从0开始）
    return x, max_index


# 随机生成一批样本
# 5类样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    counts = [0] * 5
    for a in y:
        counts[int(a.item())] += 1
        # 打印统计结果
    print(f"本次预测集中0到4维样本分别有{counts[0]}，{counts[1]}，{counts[2]}，{counts[3]}，{counts[4]}个")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # 找到预测概率最大的索引（即预测类别）
            pred_class = torch.argmax(y_p).item()
            # 真实类别（转为整数）
            true_class = int(y_t)

            if pred_class == true_class:
                correct += 1  # 预测正确
            else:
                wrong += 1  # 预测错误

        print(f"正确预测个数：{correct}, 错误预测个数：{wrong}")
        print(f"正确率：{correct / (correct + wrong):.4f}")
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 100000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.0003  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = y.squeeze(dim=1) #删除第 2 个维度，因为 dim=1
            y = y.long()   # 同时确保标签是 Long 类型（整数）
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "modelwork.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        # 1. 找到最大概率对应的类别（0-4）
        pred_class = torch.argmax(res).item()  # 最大概率的索引（类别）
        # 2. 找到最大概率值
        max_prob = torch.max(res).item()  # 最大概率的具体值
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred_class, max_prob))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # # 预测[4,2,0,0]
    # predict("modelwork.bin", test_vec)
