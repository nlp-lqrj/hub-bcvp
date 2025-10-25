import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

"""

week02作业
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个五维随机向量，值最大维度所在数组索引n，即为第n类
......

"""


# 随机生成一批样本
def build_dataset(total_sample_num):
    # 二维张量(total_sample_num, 5)
    x = torch.randn(total_sample_num, 5)
    # 一维张量(total_sample_num,)
    y = torch.argmax(x, dim=1)
    return x, y


# 定义模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 线性层，分五种类别
        self.linear = nn.Linear(input_size, 5, bias=True)
        # 交叉熵loss函数
        self.loss = nn.CrossEntropyLoss()

    # 训练时返回loss值，推理时返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 测试函数
def evaluate_model(model, test_sample_num=1000):
    model.eval()
    x, y = build_dataset(test_sample_num)
    with torch.no_grad():
        # 每个样本对应预测输出值(test_sample_num, 5)
        y_pred_logits = model(x)
        # 输出值转为类别(test_sample_num,)
        y_pred = torch.argmax(y_pred_logits, dim=1)
        # 计算准确率
        correct = (y_pred == y).sum().item()
        total = y.size(0)
        accuracy = correct / total
        print(f"测试样本数: {total}, 正确预测: {correct}, 准确率: {accuracy:.6f}")


# 训练函数
def train_model(model, optimizer, num_epochs, batch_size, train_sample_num):
    # 构建数据集
    x, y = build_dataset(train_sample_num)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 计算loss
            loss = model(batch_x, batch_y)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()
            # 累加批次loss
            total_loss += loss.item()
        print(f"=========\n第{epoch + 1}轮平均loss: {total_loss / len(train_loader):.6f}")
        # 测试
        evaluate_model(model)


if __name__ == '__main__':
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 50  # 每批训练样本个数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    total_sample_num = 5000 # 样本总数
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练
    train_model(model, optimizer, epoch_num, batch_size, total_sample_num)
