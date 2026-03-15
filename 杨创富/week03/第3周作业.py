#coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch的RNN模型
实现序列标注任务：判断文本中每个位置是否为特定字符（你/我/他）
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_dim=64):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        # RNN层：处理序列信息，捕捉位置特征
        self.rnn = nn.GRU(
            input_size=vector_dim,  # 输入维度（embedding维度）
            hidden_size=hidden_dim,  # 隐藏层维度
            num_layers=1,            # RNN层数
            batch_first=True         # 输入格式为(batch_size, seq_len, feature)
        )
        self.classify = nn.Linear(hidden_dim, 1)  # 将RNN输出映射为每个位置的预测
        self.loss = nn.BCELoss()  # 二元交叉熵损失（适合二分类序列标注）

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, seq_len) -> (batch_size, seq_len, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_dim)，获取所有时间步输出
        pred = self.classify(rnn_out)  # (batch_size, seq_len, 1)
        pred = torch.sigmoid(pred).squeeze(-1)  # (batch_size, seq_len)，归一化并去除最后一维

        if y is not None:
            return self.loss(pred, y)  # 计算损失
        else:
            return pred  # 返回预测结果


def build_vocab():
    """构建字符表"""
    chars = "你我他defghijklmnopqrstuvwxyz"  # 包含特定字符和其他字符
    vocab = {"pad": 0}  # 填充符
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 字符到索引的映射
    vocab['unk'] = len(vocab)  # 未知字符
    return vocab


def build_sample(vocab, sentence_length):
    """生成单个样本：输入序列+每个位置的标签（1表示是特定字符，0表示不是）"""
    # 随机生成长度为sentence_length的字符序列
    chars = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 转换为索引序列
    x = [vocab.get(char, vocab['unk']) for char in chars]
    # 生成标签：特定字符（你/我/他）对应位置标1，其他标0
    # 特定字符在vocab中的索引为1（你）、2（我）、3（他）
    y = [1 if idx in {1, 2, 3} else 0 for idx in x]
    return x, y


def build_dataset(sample_num, vocab, sentence_length):
    """构建数据集"""
    dataset_x = []
    dataset_y = []
    for _ in range(sample_num):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    # 转换为tensor（x为长整型，y为浮点型）
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    """构建模型"""
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sentence_length):
    """评估模型：计算位置级准确率"""
    model.eval()
    # 生成200个测试样本
    x, y = build_dataset(200, vocab, sentence_length)
    total_positions = y.numel()  # 总位置数
    correct_positions = 0  # 正确预测的位置数

    with torch.no_grad():  # 不计算梯度
        y_pred = model(x)  # 预测结果：(batch_size, seq_len)
        # 二值化预测结果（>=0.5为1，否则为0）
        y_pred_label = (y_pred >= 0.5).float()
        # 统计正确位置数
        correct_positions = (y_pred_label == y).sum().item()

    accuracy = correct_positions / total_positions
    print(f"测试集共{total_positions}个位置，正确预测{correct_positions}个，准确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 批次大小
    train_sample_num = 1000  # 每轮训练样本数
    char_dim = 30  # 字符嵌入维度
    sentence_length = 6  # 句子长度（固定）
    learning_rate = 0.001  # 学习率

    # 构建字符表
    vocab = build_vocab()
    # 构建模型
    model = build_model(vocab, char_dim, sentence_length)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练日志（记录准确率和损失）
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        # 分批次训练
        for batch in range(int(train_sample_num / batch_size)):
            # 生成批次样本
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optimizer.zero_grad()  # 梯度清零
            loss = model(x, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()

        # 每轮训练后评估
        print(f"\n第{epoch+1}轮训练，平均损失：{total_loss / (train_sample_num/batch_size):.6f}")
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, total_loss / (train_sample_num/batch_size)])

    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot([l[0] for l in log], label="准确率")
    plt.title("训练过程准确率")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot([l[1] for l in log], label="损失")
    plt.title("训练过程损失")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 保存模型和字符表
    torch.save(model.state_dict(), "rnn_model.pth")
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def predict(model_path, vocab_path, input_strings):
    """使用训练好的模型预测特定字符位置"""
    # 配置参数（需与训练时一致）
    char_dim = 30
    sentence_length = 6
    hidden_dim = 64

    # 加载字符表和模型
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = TorchModel(char_dim, sentence_length, vocab, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 处理输入
    x = []
    for s in input_strings:
        # 统一长度（截断或补全）
        if len(s) < sentence_length:
            s += "pad" * (sentence_length - len(s))  # 补全
        else:
            s = s[:sentence_length]  # 截断
        # 转换为索引
        x_idx = [vocab.get(c, vocab['unk']) for c in s]
        x.append(x_idx)

    # 预测
    with torch.no_grad():
        pred = model(torch.LongTensor(x))  # (batch_size, seq_len)
        pred_labels = (pred >= 0.5).int()  # 二值化

    # 输出结果
    for i, s in enumerate(input_strings):
        print(f"\n输入文本：{s}")
        print("位置预测结果（1表示是特定字符）：")
        for idx, (char, label, prob) in enumerate(zip(s, pred_labels[i], pred[i])):
            print(f"位置{idx}：字符'{char}'，预测{label.item()}（概率：{prob.item():.4f}）")


if __name__ == "__main__":
    main()
    # 测试预测
    test_strings = ["你abcde", "f我ghij", "kl他mno", "pqrstu", "你我他xyz", "a你b我c他"]
    predict("rnn_model.pth", "vocab.json", test_strings)