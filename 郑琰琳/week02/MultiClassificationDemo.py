import torch.nn
from torch import nn

"""
用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类
"""


class MultiClassificationModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiClassificationModule, self).__init__()
        # 线性层
        self.linear1 = nn.Linear(input_size, hidden_size)
        # GELU激活函数  GELU激活函数的输出在输入接近于 0 时接近于高斯分布，这有助于 提高神经网络的泛化能力 ，使得模型更容易适应不同的数据分布。
        self.gelu = nn.GELU()
        # 线性层
        self.linear2 = nn.Linear(hidden_size, output_size)
        # 交叉熵损失函数
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear1(x)
        x = self.gelu(x)
        y_pred = self.linear2(x)
        if y is None:
            return y_pred
        else:
            return self.cross_loss(y_pred, y)


# 准备训练数据
def build_dataSet(train_total, classification_num):
    x = torch.randn(train_total, classification_num)
    # 获取预测类别,从概率中选出最可能的类别  找出每个样本中概率最大的类别索引  argmax是根据索引位置来决定类别的，
    y = torch.argmax(x, dim=1)
    return x, y


# 模型训练
def train_model():
    # 训练轮次
    train_epoch = 200
    # 训练量
    train_total = 1000
    # 分类数
    classification_num = 5
    # 样本 张量维度
    input_size, hidden_size, output_size = 5, 10, 5
    # 构建模型
    model = MultiClassificationModule(input_size, hidden_size, output_size)
    # 学习率
    lr = 0.001
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 构建训练集
    x, y = build_dataSet(train_total, classification_num)
    # 开始训练
    for epoch in range(train_epoch):
        # 计算损失函数
        loss = model(x, y)
        # 计算梯度
        loss.backward()
        # 更新权重
        optimizer.step()
        optimizer.zero_grad()
        # 每批次的loss
        print('Epoch {}, Loss: {:.6f}'.format(epoch+1, loss.item()))

    torch.save(model.state_dict(), '../../model/week02/multi_classification_model.pth')


# 模型验证
def predict_model(input_vec):
    input_size, hidden_size, output_size = 5, 10, 5
    # 构建模型
    model = MultiClassificationModule(input_size, hidden_size, output_size)
    # 加载本地模型
    model.load_state_dict(torch.load('../../model/week02/multi_classification_model.pth'))
    model.eval()
    # 不计算提的
    with torch.no_grad():
        # 模型预测
        logits = model.forward(input_vec)
        # softmax 归一化 最大的概率
        probabilities = torch.softmax(logits, dim=1)
        # argmax 最大值的下标
        predictions = torch.argmax(logits, dim=1)
    for inp, prob, pred in zip(input_vec, probabilities, predictions):
        # 真实值
        actual_class = torch.argmax(inp)
        print(f"输入：{inp}，"
              f"真实值：{actual_class},"
              f"预测值：{pred},"
              f"预测概率：{prob}")


# 测试
if __name__ == '__main__':
    train_model()
    test_vec = torch.randn(4, 5)
    print(test_vec)
    predict_model(test_vec)

