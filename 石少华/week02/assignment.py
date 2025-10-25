import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

'''
作业内容：
用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类
'''

'''
获取数据的标签
一组数据共有五个特征，对应类别为特征值最大的那个类
'''
def build_sample():
    x = np.random.random(5)
    label = np.argmax(x)
    return x, label


'''
根据需要生成的样本数生成对应的样本
'''
def generate_dataset(num_samples):
    X = []
    Y = []
    for i in range(num_samples):
        x, y = build_sample()
        X.append(x)
        Y.append(y)

    return X, Y

'''
定义模型
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Linear(5, 5)

    def forward(self, x):
        y = self.layer(x)
        return y


# 定义损失函数
loss = nn.CrossEntropyLoss()

# 定义超参数
num_samples = 1000
learning_rate = 0.001
batch_size = 20
epochs = 100

# 获取并封装数据
features, labels = generate_dataset(num_samples)

trainset = TensorDataset(
    torch.Tensor(features[:800]),
             torch.tensor(labels[:800], dtype=torch.long)
             )
testset = TensorDataset(
    torch.Tensor(features[800:]),
            torch.tensor(labels[800:], dtype=torch.long)
            )

train_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    testset,
    batch_size=len(testset),
    shuffle=False
)

# 定义计算网络
net = Net()

# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# 评估模型在测试集上的准确率
def evaluate():
    correct = 0
    for x, y in test_loader:
        y_pred = net.forward(x)
        # 使用softmax进行归一化
        y_pred = nn.Softmax(dim=1)(x)
        y_pred = torch.argmax(y_pred, dim=1)
        correct += (y_pred == y).sum().item()

    print(f"accuracy is {correct / len(testset)}")

if __name__ == '__main__':
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        for x, y in train_loader:
            y_pred = net.forward(x)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            _, predicted = torch.max(y_pred, 1)

            running_loss += l.item()
            correct += (y == predicted).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / 800
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f}")

    print('-------测试集上的数据评估----------')
    evaluate()
