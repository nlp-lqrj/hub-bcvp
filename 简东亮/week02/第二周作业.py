# 解决 OpenMP 库冲突问题
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 128)  # 增加隐藏层
        self.linear2 = nn.Linear(128, num_classes)  # 输出层，5个类别
        self.activation = torch.relu  # 使用ReLU激活函数
        self.loss = nn.functional.cross_entropy  # 使用交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 128)
        x = self.activation(x)  # 使用ReLU激活
        x = self.linear2(x)  # (batch_size, 128) -> (batch_size, num_classes)

        if y is not None:
            return self.loss(x, y)  # 交叉熵损失直接使用logits
        else:
            return torch.softmax(x, dim=1)  # 预测时返回softmax概率


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，找出最大值所在的维度索引作为类别标签
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)  # 找出最大值所在的索引
    return x, max_index

# 随机生成一批样本
def build_dataset(total_sample_num):    # 预分配numpy数组
    X = np.zeros((total_sample_num, 5))
    Y = np.zeros(total_sample_num, dtype=np.int64)

    for i in range(total_sample_num):
        x, y = build_sample()
        X[i] = x  # 直接赋值到预分配的数组
        Y[i] = y

    # 将标签转换为LongTensor类型，因为交叉熵损失需要Long类型
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计每个类别的样本数量
    class_count = [0] * 5
    for label in y:
        class_count[label] += 1
    print("各类别样本数量:", class_count)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，返回softmax概率
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测类别

        for y_p, y_t in zip(predicted_classes, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 200  # 增加训练轮数
    batch_size = 20  # 批量大小
    train_sample = 1000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数量
    learning_rate = 0.002  # 调整学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据
            start_idx = batch_index * batch_size
            end_idx = (batch_index + 1) * batch_size
            x = train_x[start_idx:end_idx]
            y = train_y[start_idx:end_idx]

            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "multiclass_model.bin")

    # 画图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(log)), [l[0] for l in log], label="accuracy",color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(log)), [l[1] for l in log], label="loss", color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测

    for vec, res in zip(input_vec, result):
        predicted_class = torch.argmax(res).item()
        confidence = res[predicted_class].item()
        true_class = np.argmax(vec)
        print("输入：%s, 真实类别：%d, 预测类别：%d, 置信度：%f" %
              (vec, true_class, predicted_class, confidence))


if __name__ == "__main__":
    main()

    # 测试预测
    test_vec = [
        [0.9, 0.1, 0.2, 0.3, 0.4],  # 最大值在第0维
        [0.1, 0.95, 0.2, 0.3, 0.4],  # 最大值在第1维
        [0.1, 0.2, 0.98, 0.3, 0.4],  # 最大值在第2维
        [0.1, 0.2, 0.3, 0.97, 0.4],  # 最大值在第3维
        [0.1, 0.2, 0.3, 0.4, 0.99]  # 最大值在第4维
    ]
    predict("multiclass_model.bin", test_vec)
