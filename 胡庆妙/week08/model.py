# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

"""
建立网络模型结构
"""


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        embed_size = config["embed_size"]
        vocab_size = config["vocab_size"]
        sentence_len = config["sentence_len"]

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # self.lstm = nn.LSTM(embed_size, embed_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(0.5)

    # 输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        # 使用lstm
        # x, _ = self.lstm(x)

        # shape: [batch_size, sentence_len, embed_size]
        x = self.layer(x)

        # [batch_size, sentence_len, embed_size] -> [batch_size, embed_size]
        return nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze(-1)


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = self.cosine_triplet_loss

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        """
        Args:
            tensor1: shape: [batch_size, embed_size]
            tensor2: shape: [batch_size, embed_size]
        Returns:
            shape: [batch_size]
        """
        # L2归一化，即：将-1维的每个元素 a[i]/L2范数 = a[i]/sqrt(sum(a[i]^2))
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)

        # 计算余弦相似度 axb/|a|*|b|，即两个矩阵归一化后的哈达玛积
        cosine = torch.sum(torch.mul(tensor1, tensor2), dim=-1)
        return 1 - cosine

    #  三元组损失函数如下：𝐿=max( d(a,p)-d(a,n)+margin, 0)，
    #  a: anchor 原点，p: positive 与a同一类别的样本，n: negative 与a不同类别的样本
    def cosine_triplet_loss(self, a, p, n, margin=None):
        """
        Args:
            a: 原点 [batch_size, sen_len, embed_size]
            p: 正样本 [batch_size, sen_len, embed_size]
            n: 负样本 [batch_size, sen_len, embed_size]
            margin:
        """
        ap = self.cosine_distance(a, p)  # 计算a和p的余弦距离
        an = self.cosine_distance(a, n)  # 计算a和n的余弦距离
        # 如果没有设置margin，则设置diff为ap - an + 0.1
        if margin is None:
            diff = ap - an + 0.1  # [batch_size]
        # 如果设置了margin，则设置diff为ap - an + margin.squeeze(-1)
        else:
            diff = ap - an + margin.squeeze(-1)  # [batch_size]
        return torch.mean(torch.clamp(diff, min=0))  # 将小于0的元素设为0，再计算这批loss的平均值

    def forward(self, sentence_a, sentence_p=None, sentence_n=None):
        # 同时传入三个句子
        if sentence_n is not None:
            vector_a = self.sentence_encoder(sentence_a)
            vector_p = self.sentence_encoder(sentence_p)
            vector_n = self.sentence_encoder(sentence_n)
            return self.loss(vector_a, vector_p, vector_n)  # 标量

        # 单独传入一个句子时，认为正在使用向量化能力
        elif sentence_p is None and sentence_n is None:
            return self.sentence_encoder(sentence_a)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 10
    Config["sentence_len"] = 4
    model = SiameseNetwork(Config)
    a = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    p = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    n = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    y = model(a, p, n)
    print(y)
    # print(model.state_dict())
