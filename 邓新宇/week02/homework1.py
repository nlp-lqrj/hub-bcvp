import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，5维向量中最大的数字在哪维就属于哪一类

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)#线性层
        self.loss = nn.CrossEntropyLoss()#交叉熵损失函数

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y.long())
        else:
            return y_pred

#生成一个样本，随机生成一个5维向量，5维向量中最大的数字在哪维就属于哪一类
def build_sample():
    x = np.random.random(5)
    max_idx = np.argmax(x)
    label = max_idx
    return x, label

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    X_np = np.array(X)  # 形状：(total_sample_num, 5)
    Y_np = np.array(Y)  # 形状：(total_sample_num,)

    return torch.FloatTensor(X_np), torch.LongTensor(Y_np)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)  # y是真实类别索引（0-4），x是真实五维数据
    with torch.no_grad():
        y_pred = model(x)  # 模型输出logits（未归一化的分数）
        # 将logits转换为概率分布（用softmax归一化，确保总和为1）
        prob_dist = torch.softmax(y_pred, dim=1)  # 形状：(100, 5)，每个元素是对应类别的概率
        # 计算预测类别索引
        pred_indices = y_pred.argmax(dim=1)  # 形状：(100,)

        # 统计正确率（保留原有逻辑）
        correct = (pred_indices == y).sum().item()
        accuracy = correct / test_sample_num

        # 打印每个样本的真实五维数据、真实类别、概率分布和预测类别（前5个示例）
        print("\n部分样本的预测详情（真实五维数据 | 真实类别 | 概率分布 | 预测类别）：")
        for i in range(min(5, test_sample_num)):  # 只打印前5个，避免输出过长
            true_data = x[i].numpy()  # 第i个样本的真实五维数据（转为numpy数组）
            true_label = y[i].item()  # 真实类别（0-4）
            probs = prob_dist[i].numpy()  # 当前样本的5类概率
            pred_label = pred_indices[i].item()  # 预测类别（0-4）

            # 格式化打印：所有数值保留3位小数，保持格式统一
            print(
                f"样本{i + 1}：数据=[{true_data[0]:.3f}, {true_data[1]:.3f}, {true_data[2]:.3f}, {true_data[3]:.3f}, {true_data[4]:.3f}] "
                f"| 真实={true_label} "
                f"| 概率=[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}, {probs[3]:.3f}, {probs[4]:.3f}] "
                f"| 预测={pred_label}"
            )

    print(f"\n正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy

def main():
    epoch_num = 50    # 训练轮数
    batch_size = 20    # 每次训练样本个数
    train_sample = 5000    # 每轮训练总共训练的样本总数
    input_size = 5    # 输入向量维度
    learning_rate = 0.01    # 学习率
    #建立模型
    model = TorchModel(input_size)
    #选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y)    #计算Loss
            loss.backward()    #计算梯度
            optimizer.step()    #更新权重
            optimizer.zero_grad()    #梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        accuracy = evaluate(model)
        log.append([accuracy, float(np.mean(watch_loss))])
    #保存模型
    torch.save(model.state_dict(), 'model.bin')
    #画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="accuracy")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()





