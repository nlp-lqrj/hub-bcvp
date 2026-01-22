# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.classify = nn.Linear(隐藏维度 * 2, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # 损失函数采用交叉熵

    # 当输入真实标签时返回损失值；不输入真实标签时返回预测结果
    def forward(self, x, target=None):
        x = self.embedding(x)  # 输入形状：(批大小, 序列长度)
        x, _ = self.layer(x)      # 输出形状：(批大小, 序列长度, 隐藏维度 * 2)
        predict = self.classify(x) # 输出：(批大小, 序列长度, 标签数) -> (批大小 * 序列长度, 标签数)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) 
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # 形状：(样本数, 类别数)，(样本数)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)