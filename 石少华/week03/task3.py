import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import random
import string
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子，保证结果可以复现
torch.manual_seed(0)
random.seed(0)


class Model(nn.Module):
    def __init__(self, input_str, vocab_dims, hidden_dims):
        super(Model, self).__init__()
        self.str = input_str
        # 构建词表
        self.vocab = build_vocab(input_str)
        # 定义嵌入层
        self.embedding = nn.Embedding(len(input_str)+2, vocab_dims) # +2是因为有pad和unk字符
        # 定义RNN模型
        self.rnn = nn.GRU(
            vocab_dims,
            hidden_dims,
            batch_first = True
        )
        # 定义线性模型，用于获得类别
        # 输出结果应该为每个类别下对应的该样本的概率
        self.linear = nn.Linear(hidden_dims, len(input_str) + 2) # +2是因为有pad和unk字符

    def forward(self, x):
        x = torch.tensor([self.vocab.get(ch, self.vocab["UNK"]) for ch in x], dtype = torch.int32)
        x = self.embedding(x)
        x = self.rnn(x)[0]
        out = self.linear(x)
        return out

    def predict(self, input_strs, labels, criterion):
        self.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_accuracy = 0.0
            for input_str, label in zip(input_strs, labels):
                y = torch.tensor(label, dtype = torch.long)
                y_prob = self.forward(input_str)

                # 计算损失值
                total_loss += criterion(y_prob, y).item()

                # 计算准确率
                y_test_pred = torch.argmax(y_prob, dim = 1)
                total_accuracy += (y_test_pred == y).float().mean().item()

            return total_loss / len(input_strs), total_accuracy / len(input_strs)

    # 对任意输入进行测试
    def predict_str(self, str):
        y_pred = self.forward(str)
        y_indices = torch.argmax(y_pred, dim = 1)

        # 取原字符串对应位置的字符
        origin = "".join([self.str[idx - 1] for idx in y_indices])
        print(origin)



'''
    构建给定句中所有字符的位置索引表
    :return 返回位置索引表
'''
def build_vocab(str):
    if str == "":
        print("当前词表为空")
        return

    # 构建词表
    vocab = {}

    # 设定空白字符
    vocab["pad"] = 0
    index = 1
    for letter in str:
        vocab[letter] = index
        index += 1

    # 设置未知字符
    vocab["UNK"] = index
    return vocab

def build_sample(str, vocab, min_length = 3, max_length = 10):
    # 定义模型输入序列长度的最大值和最小值
    length = random.randint(min_length, max_length)
    # 随机从词表中选取length个字符，可能重复
    x = "".join([random.choice(list(string.ascii_lowercase)) for _ in range(length)])
    # 获取对应的标签
    y = [vocab.get(ch, vocab["UNK"]) for ch in x]
    # 组合数据
    sample = [x, y]

    return sample


def build_dataset(num_samples, str, vocab, min_length, max_length):
    # 多次构建样本数据，形成数据集
    feature = []
    labels = []
    for _ in range(num_samples):
        x, y = build_sample(str, vocab, min_length, max_length)
        feature.append(x)
        labels.append(y)

    return feature, labels


if __name__ == "__main__":
    # 定义目标字符串
    s = "dcbopstyxrqefglnhijkuvw"

    # 构建序列与索引字典
    vocab = build_vocab(s)

    # 定义数据集参数
    n_samples = 1000
    min_length = 3
    max_length = 10

    # 构建数据集
    x, y = build_dataset(n_samples, s, vocab, min_length, max_length)
    # 训练集和测试集分割
    X_train, X_test, y_train, y_test = train_test_split(
        x, y,
        train_size = 0.9,
        shuffle = True,
        random_state = 42
    )

    # 定义模型参数
    vocab_dims = 5
    hidden_dims = 20
    lr = 0.01
    num_epochs = 20

    # 定义模型，损失函数，优化器
    model = Model(s, vocab_dims, hidden_dims)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    acc_history = []
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        for x, y in zip(X_train, y_train):
            y = torch.tensor(y, dtype = torch.long)
            # 计算出来的是字符位于当前索引的概率
            logits = model.forward(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 添加损失
            total_loss += loss.item()
            # 计算准确率
            y_pred = torch.argmax(logits, dim = 1)
            total_acc += (y_pred == y).float().mean().item()
        avg_test_loss, avg_test_acc = model.predict(X_test, y_test, criterion)

        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {total_loss/len(X_train):.4f}, Train Acc: {total_acc/len(X_train):.4f} | "
              f"Test Loss: {avg_test_loss}, Test Acc: {avg_test_acc}")

    # 对任意字符串进行测试
    model.predict_str("hello")





