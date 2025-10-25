# Week02

### 任务

实现一个五分类任务，x是一个5维向量，最大数字在第几维就属于第几类，使用交叉熵作为Loss。

### 代码

```python
# coding:utf8

# 解决 OpenMP 库冲突问题
# 黄鸿和 TripleH  2025.10.24


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
实现一个五分类任务
规律：x是一个5维向量，最大数字在第几维就属于第几类（0-4类）
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层，输出5个类别
        self.activation = torch.softmax  # softmax归一化函数，用于多分类
        self.loss = nn.functional.cross_entropy  # 交叉熵损失函数，适用于多分类

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            # 对于交叉熵损失，不需要softmax
            return self.loss(logits, y.long().squeeze())  # 预测值和真实值计算损失
        else:
            # 预测时使用softmax归一化
            y_pred = self.activation(logits, dim=1)  # (batch_size, num_classes) -> (batch_size, num_classes)
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大数字在第几维就标记为第几类
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)  # 找到最大值的索引
    return x, max_index


# 随机生成一批样本
# 五个类别均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, show_samples=False):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    # 统计每个类别的样本数量
    class_counts = [0] * 5
    for label in y:
        class_counts[label.item()] += 1
    print("本次预测集中各类别样本数量：", class_counts)
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        
        # 如果需要显示样本详情
        if show_samples:
            print("========= 测试样本详情 =========")
            for i in range(min(10, len(x))):  # 只显示前10个样本
                sample = x[i].numpy()
                true_class = y[i].item()
                pred_class = predicted_classes[i].item()
                pred_prob = y_pred[i].numpy()
                max_idx = np.argmax(sample)
                status = "✓" if true_class == pred_class else "✗"
                print("样本 %d: %s -> 真实类别: %d, 预测类别: %d, 预测概率: %s %s" % 
                      (i+1, sample, true_class, pred_class, pred_prob, status))
            print("================================")
        
        for pred, true_label in zip(predicted_classes, y):
            if pred.item() == true_label.item():
                correct += 1
            else:
                wrong += 1
    
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 分类数量
    learning_rate = 0.001  # 学习率
    
    # 建立模型（5 * 5）
    model = TorchModel(input_size, num_classes)
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
            #取出一个batch数据作为输入
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            
            # 输出每个样本的类别信息（只在第一个epoch的第一个batch显示）
            if epoch == 0 and batch_index == 0:
                print("========= 样本类别信息 =========")
                for i in range(min(10, len(x))):  # 只显示前10个样本
                    sample = x[i].numpy()
                    true_class = y[i].item()
                    max_idx = np.argmax(sample)
                    print("样本 %d: %s -> 真实类别: %d (最大值在第%d维)" % 
                          (i+1, sample, true_class, max_idx))
                print("================================")
            
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 最后一轮显示测试样本详情
        show_samples = (epoch == epoch_num - 1)
        acc = evaluate(model, show_samples)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    
    # 保存模型
    torch.save(model.state_dict(), "model_five_class.bin")
    
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
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        predicted_classes = torch.argmax(result, dim=1)  # 获取预测的类别
    
    for vec, pred_class, prob in zip(input_vec, predicted_classes, result):
        max_idx = np.argmax(vec)
        print("输入：%s, 真实类别：%d, 预测类别：%d, 预测概率：%s" % 
              (vec, max_idx, pred_class.item(), prob.numpy()))


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model_five_class.bin", test_vec)

```

### 效果展示

#### 训练过程

![image-20251024141938462](week02.assets/image-20251024141938462.png)

![image-20251024141953331](week02.assets/image-20251024141953331.png)

#### Loss图和准确率图

![image-20251024141809245](week02.assets/image-20251024141809245.png)