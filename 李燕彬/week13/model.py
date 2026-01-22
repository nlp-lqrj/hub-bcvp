# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel, AutoModelForTokenClassification  # 导入BERT模型
"""
建立网络模型结构
"""

# 直接使用AutoModelForTokenClassification作为基础模型
from config import Config

# 创建基础模型
TorchModel = AutoModelForTokenClassification.from_pretrained(
    Config["bert_path"],
    num_labels=Config["class_num"]
)

# 保持原有的TorchModel类定义，用于兼容旧代码
class TorchModelOld(nn.Module):
    def __init__(self, config):
        super(TorchModelOld, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1 if "vocab_size" in config else 0
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        
        # 检查是否使用BERT模型
        self.use_bert = config["use_bert"]
        
        if self.use_bert:
            # 使用BERT作为特征提取器
            self.bert = BertModel.from_pretrained(config["bert_path"])
            # BERT的隐藏层大小
            bert_hidden_size = self.bert.config.hidden_size
            # 添加分类层，将BERT输出映射到标签空间
            self.classify = nn.Linear(bert_hidden_size, class_num)
        else:
            # 原有模型结构
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
            self.classify = nn.Linear(hidden_size * 2, class_num)
        
        # 损失函数
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if self.use_bert:
            # 使用BERT模型获取嵌入表示
            outputs = self.bert(x)
            x = outputs[0]  # shape:(batch_size, sen_len, bert_hidden_size) - 使用索引访问兼容不同版本transformers库
        else:
            # 原有LSTM特征提取路径
            x = self.embedding(x)  # input shape:(batch_size, sen_len)
            x, _ = self.layer(x)      # input shape:(batch_size, sen_len, hidden_size * 2)
        
        predict = self.classify(x)  # output:(batch_size, sen_len, num_tags)

        if target is not None:
            #(number, class_num), (number)
            return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    
    if hasattr(model, "use_bert") and model.use_bert:
        # 对BERT模型进行参数分组，预训练参数使用较小的学习率
        bert_params = list(model.bert.named_parameters())
        other_params = list(model.classify.named_parameters())
        
        # 设置不同参数组的学习率
        param_groups = [
            {'params': [p for n, p in bert_params], 'lr': learning_rate * 0.1},  # BERT参数学习率×0.1
            {'params': [p for n, p in other_params], 'lr': learning_rate}  # 其他参数使用正常学习率
        ]
        
        if optimizer == "adam":
            return Adam(param_groups)
        elif optimizer == "sgd":
            return SGD(param_groups)
    else:
        # 原有优化器设置
        if optimizer == "adam":
            return Adam(model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
