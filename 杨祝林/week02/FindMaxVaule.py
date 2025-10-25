# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

"""


class MyTorchModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_classes=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, num_classes)  # 输出层（5个神经元，输出logits）
        )
        self.loss = nn.CrossEntropyLoss()  # 内置softmax，直接接收数据和标签


    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.layers(x)  # (batch_size, input_size) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果




# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，找出其中的最大数的下标
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        # 取logits最大值对应的索引作为预测类别
        predicted_classes = torch.argmax(y_pred, dim=1)  # 形状: (1000,)
        # 比较预测类别与真实标签
        correct = (predicted_classes == y).sum().item()
    accuracy = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy



def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = MyTorchModel(input_size, 64, 5)

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
            loss = model.forward(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "mymodel.pt")
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
    model = MyTorchModel(input_size, 64, 5)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        # 计算概率（可选，用于查看置信度）
        probabilities = torch.softmax(result, dim=1)
        # 取最大概率对应的类别
        predicted_classes = torch.argmax(result, dim=1)

    # 打印结果
    for vec, cls, prob in zip(input_vec, predicted_classes, probabilities):
        max_prob = prob[cls].item()  # 该类别的概率
        print(f"输入：{vec}, 预测最大元素所在维度：{cls.item()}, 置信度：{max_prob:.4f}")



if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("mymodel.pt", test_vec)
