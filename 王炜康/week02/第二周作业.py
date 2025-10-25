import os
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


#. 定义模型
class TorchModel(nn.Module):
    def __init__(self, input_size, n_class):
        super(TorchModel, self).__init__()
        self.liner1 = nn.Linear(input_size, 64)
        self.activate = nn.Sigmoid()
        self.liner2 = nn.Linear(64, n_class)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x, y=None):
        x = self.liner1(x)
        x = self.activate(x)
        y_pred = self.liner2(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
        
# 生成一个样本，哪个值最大属于哪一类。
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码:
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x).argmax(1)
        for y_p, y_t in zip(y_pred, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print(f'正确预测个数：{correct}, 正确率：{correct/(correct+wrong)}')
    return correct/(correct+wrong)


def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    n_class = 5
    learning_rate = 0.001 
    model = TorchModel(input_size, n_class)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    #创建训练集
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index+1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index+1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print(f'==========\n第{epoch + 1}轮平均loss：{np.mean(watch_loss)}')
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
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
    n_class = 5
    model = TorchModel(input_size, n_class)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果
        print(f'输入：{vec}, 预测类别：{int(res.argmax())}, 概率值：{nn.functional.softmax(res).max()}')


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
        





        
        
