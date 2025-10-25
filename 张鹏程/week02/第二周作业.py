# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务、
改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
规律：x是一个5维向量，学习判断每组最大数下标
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失，包含Softmax激活函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 输出预测结果，转换为概率（5维）


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量
def build_sample():
    x = np.random.random(5)
    idx = np.argmax(x)
    return x, idx


# 随机生成一批样本
def build_dataset(total_sample_num):
    x = []
    y = []
    for i in range(total_sample_num):
        x1, y1 = build_sample()
        x.append(x1)
        y.append(y1)
    return torch.FloatTensor(x), torch.LongTensor(y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():  # 不计算梯度
        y_pred_logits = model(x)  # 模型输出logits（5维）
        y_pred = torch.argmax(y_pred_logits, dim=1)  # 取概率最大的索引作为预测结果
        # 统计正确个数
        correct = (y_pred == y).sum().item()
    accuracy = correct / test_sample_num
    print(f"测试集总样本：{test_sample_num}个，正确预测：{correct}个，准确率：{accuracy:.4f}\n")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 总共训练的样本总数
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
        total_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            # print('----->loss\n', loss)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            total_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(total_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(total_loss))])
        if acc > 0.9999:
            break
    # 保存模型
    torch.save(model.state_dict(), "mc.bin")
    # 画图
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
    # print(model.state_dict())
    model.eval()  # 测试模式

    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测
        y_pred_probs = torch.softmax(result, dim=1)  # 转换为概率（5维）
        y_pred = torch.argmax(y_pred_probs, dim=1)  # 预测索引
        # 输出结果
        for vec, probs, pred_idx in zip(input_vec, y_pred_probs, y_pred):
            true_idx = np.argmax(vec)  # 真实最大值索引
            max_val = vec[true_idx]  # 真实最大值
            print(f"输入：{vec}")
            print(f"最大索引：{true_idx}，值：{max_val}")
            print(f"预测索引：{pred_idx.item()}，预测概率：{probs[pred_idx].item():.4f}\n")


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
    #             [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
    #             [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
    #             [0.98949776, 0.9991666, 0.92579291, 0.41567412, 0.1358894]]
    # predict("mc.bin", test_vec)
