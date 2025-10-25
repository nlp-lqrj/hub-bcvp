# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果x[i]最大，就属于i类别
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        x = self.linear1(x)
        y_pred = self.linear2(x)
        if y is not None:
            return self.loss(y_pred, y)
        return torch.softmax(y_pred, dim = 1)
    
# 生成一组数据
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)

# 生成样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 500
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        y_class = torch.argmax(y_pred, dim = 1)
        correct += (y_class ==y).sum().item()
    print("正确率为：%f" % (correct / test_sample_num))
    return correct / test_sample_num

def main():
    epoch_num = 100
    batch_size = 20
    train_sample_num = 50000
    input_size = 5
    hidden_size = 10
    output_size = 5
    learning_rate = 0.001
    model = TorchModel(input_size, hidden_size, output_size)
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample_num)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for index in range(train_sample_num // batch_size):
            x = train_x[index * batch_size : (index + 1) * batch_size]
            y = train_y[index * batch_size : (index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮测试平均loss：%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
        if acc > 0.9999:
            break
    
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    plt.plot(range(len(log)), [l[0] for l in log], label = "acc")
    plt.plot(range(len(log)), [l[1] for l in log], label = "loss")
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    hidden_size = 10
    output_size = 5
    model = TorchModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    # print(model.state_dict())

    model.eval()
    with torch.no_grad():    # 不计算梯度
        result = model(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d， 概率值：%f" % (vec, np.argmax(res), res[np.argmax(res)]))

if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
