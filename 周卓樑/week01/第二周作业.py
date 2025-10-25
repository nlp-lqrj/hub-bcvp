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
规律：x是一个5维向量，比较哪一个数更大，给出分类标号

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=1) #将y_pred的训练集softmax归一化


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，输出哪一个位置的数值最大y
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x , y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        pred_classes = torch.argmax(y_pred, dim=1)
        for y_p, y_t in zip(pred_classes, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
        acc = correct / test_sample_num
    print(f"测试集准确率: {acc:.4f}")
    return acc

def main():
    # 配置参数
    epoch_num = 50 # 训练轮数
    batch_size = 50  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
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
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        # print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(f"第{epoch + 1}轮平均loss: {np.mean(watch_loss):.4f}")
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
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
        pred_class = torch.argmax(result, dim=1)
    for vec, res, cls in zip(input_vec, result, pred_class):
        print(f"输入: {vec} → 预测类别: {cls.item()+1} 概率分布: {res.numpy()}")

if __name__ == "__main__":
    main()
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95558807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)



// 实际运行第49轮平均loss: 0.2123
// 测试集准确率: 0.9900
// 第50轮平均loss: 0.2104
// 测试集准确率: 0.9900
// 测试数据运行结果
// 输入: [0.88889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843] → 预测类别: 1 概率分布: [5.0497180e-01 6.4799900e-07 1.2061981e-05 6.7071412e-08 4.9501544e-01]
// 输入: [0.74963533, 0.5524256, 0.95558807, 0.95520434, 0.84890681] → 预测类别: 3 概率分布: [1.0203838e-02 2.6738370e-04 4.9751869e-01 4.2656454e-01 6.5445557e-02]
// 输入: [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392] → 预测类别: 1 概率分布: [9.8666787e-01 1.3297411e-02 7.1816527e-07 3.1751264e-05 2.2364179e-06]
// 输入: [0.99349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894] → 预测类别: 1 概率分布: [7.6502842e-01 4.7748385e-04 2.3447715e-01 1.6780235e-05 1.1154809e-07]
