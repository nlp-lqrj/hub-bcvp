# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class NerClassificationModel(nn.Module):
    def __init__(self, config):
        super(NerClassificationModel, self).__init__()
        # 参数
        self.label_num = config["label_num"]
        self.use_crf = config["use_crf"]
        self.dropout_rate = config["dropout"]
        self.model_path = config["pretrain_model_path"]

        self.encoder = BertModel.from_pretrained(self.model_path, return_dict=True)
        # PEFT必须要获取model中的config参数，所以需要显式定义
        self.config = self.encoder.config
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, self.label_num)

        self.crf = CRF(self.label_num, batch_first=True)

    # 当与peft适配时，入参名称必须严格按照transformer框架的参数定义，即bert入参名必须为input_ids
    # 当peft解析forward函数时，需要按照transformer中的所有参数名称进行一一解析，所以需要传入**kwargs供其解析
    def forward(self, input_ids, tags=None, **kwargs):
        x = self.encoder(input_ids).last_hidden_state
        emissions = self.classifier(self.dropout(x))   # (batch_size, seq_len, label_num)

        # labels为空，进行预测
        if tags is None:
            if self.use_crf:
                # 使用crf.decode解码，返回batch中每个样本的最优的标签序列列表（batch_size, seq_len）
                return self.crf.decode(emissions)
            return torch.softmax(emissions, dim=-1)
        # 传入labels，计算loss
        else:
            if self.use_crf:
                # 使用crf计算loss时，传入预测值和真实值（mask可选），最后一定要加负号
                mask = tags != -1
                return -self.crf(emissions, tags, mask)
            else:
                loss = nn.CrossEntropyLoss(ignore_index=-1)
                # 序列标注任务中，预测值(batch_size, seq_len, label_num)不能与真实标签(batch_size, seq_len)直接计算loss
                # 需要转化成：预测值（batch_size * seq_len, label_num），真实标签（batch_size * seq_len）
                emissions = emissions.view(-1, emissions.shape[-1])
                labels = tags.veiw(-1)   # 等效于 tags.flatten()，但是效率比view低
                return loss(emissions, labels)




