import numpy as np
import torch.nn
from torch import nn

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本
"""

class BinaryClassificationModel(nn.Module):
    # 初始化
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryClassificationModel, self).__init__()
        # 线性层
        self.linear = nn.Linear(input_size, hidden_size)
        # GELU激活函数
        self.gelu = nn.GELU()
        # 线性层
        self.linear2 = nn.Linear(hidden_size, output_size)
        # 交叉熵损失函数 包含softmax
        self.loss = nn.CrossEntropyLoss()

    # 定义模型
    def forward(self, x, y=None):
        x = self.linear(x)
        x = self.gelu(x)
        y_pre = self.linear2(x)
        if y is None:
            return y_pre
        else:
            if y.dim() > 1:
                # 移除张量中所有维度为1的维度（压缩维度） 将 [[1], [0], [1], [0]] 变成 [1, 0, 1, 0]
                y = y.squeeze()
            if y.dtype != torch.long:
                # 将张量的数据类型转换为长整型（int64） 将 [1.0, 0.0, 1.0, 0.0] 变成 [1, 0, 1, 0]
                y = y.long()
            return self.loss(y_pre, y)


# 生成样本，如果第一个数值大于第五个返回1，否则返回0
def build_sample():
    # x = torch.randn(1, 5)
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1  # 正样本
    else:
        return x, 0  # 负样本


# 数据集准备
def build_dataSet(train_total):
    X = []
    Y = []
    for i in range(train_total):
        x, y = build_sample()
        X.append(x)
        Y.append(y)   # 直接存储标量，不是列表
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 模型训练
def trainModel():
    # 训练轮次
    epochs = 20
    # 训练集总数
    train_total = 5000
    # 每次训练数量
    batch_size = 20
    # 输入
    input_size = 5
    hidden_size = 10
    # 输出 二分类：两个类别
    output_size = 2

    # 建立模型
    model = BinaryClassificationModel(input_size, hidden_size, output_size)
    # 学习率定义
    lr = 0.001
    # 优化器定义
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # 训练集构建
    train_x, train_y = build_dataSet(train_total)

    # 开始训练
    for epoch in range(epochs):
        model.train(True)
        epoch_loss = []
        for batch_index in range(train_total // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]

            # 计算损失函数
            loss = model(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度归0
            optim.zero_grad()

            # 存储 loss
            epoch_loss.append(loss.item())

        print("第 {} 轮，平均Loss {:.6f}".format(epoch+1, np.mean(epoch_loss)))   # 保留6位小数：

    # 保存模型
    torch.save(model.state_dict(), '../../model/week02/binary_classification_model.bin')
    # 画图
    return model

# 验证模型
def predict(model_path, input_vec):
    input_size = 5
    hidden_size = 10
    output_size = 2
    # 构建模型
    model = BinaryClassificationModel(input_size, hidden_size, output_size)
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    # 测试模型
    model.eval()

    # 不计算梯度
    with torch.no_grad():
        # 原始输出
        logits = model.forward(torch.FloatTensor(input_vec))
        # 使用softmax 获取概率 每个类别的可能性是多少  将模型的原始输出转换为概率分布  把 [2.0, 1.0] 这样的原始分数变成 [0.7311, 0.2689] 这样的概率值，所有类别概率加起来等于1。
        probabilities = torch.softmax(logits, dim=1)  # 给人看的，显示每个类别的置信度
        # 获取预测类别,从概率中选出最可能的类别  找出每个样本中概率最大的类别索引  argmax是根据索引位置来决定类别的，  样本1：[0.7311, 0.2689] → 类别0（因为0.7311 > 0.2689） 样本2：[0.1824, 0.8176] → 类别1（因为0.8176 > 0.1824）
        predictions = torch.argmax(logits, dim=1)  # 给程序用的，直接得到最终预测结果

    for inp, prob, pred in zip(input_vec, probabilities, predictions):
        # 真实值
        actual_class = 1 if inp[0] > inp[4] else 0  # python 三元表达式
        correct_class = "✓" if pred == actual_class else "✗"
        print(f"输入：{[f'{val:.6f}' for val in inp]},"
              f"实际：{actual_class},预测：{pred},"   # pred.item() 从张量中提取出预测的数值
              f"概率：[0:{prob[0]:.4f},{prob[1]:.4f}] {correct_class}")


# 验证数据
if __name__ == '__main__':
    # 训练
    trainModel()
    # 构建验证集
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # 验证
    predict('../../model/week02/binary_classification_model.bin', test_vec)
