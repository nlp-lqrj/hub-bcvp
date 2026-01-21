# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        # vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.bert_like = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        hidden_size = self.bert_like.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        # x, _ = self.layer(x)      #input shape:(batch_size, sen_len, hidden_size * 2)
        x, _ = self.bert_like(x)  # (batch_size, sen_len) -> (batch_size, sen_len, hidden_size),  (batch_size, hidden_size)
        predict = self.classify(x) # output:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                # pytorch-crf 库返回的计算结果是未取相反数的S(X, y) - log(Z(X))，这里需要取反才是loss = log(Z(X)) - S(X, y
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                # 交叉熵对输入有形状要求(number, class_num), (number)，需要view转变张量形状
                # (batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)  # 解码出来的是一个序列 (batch_size, sen_len) （在evaluate的时候要想办法把标签序列还原为文本段）
            else:
                return predict   # （batch_size, sen_len, num_tags)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 20
    model = TorchModel(Config)