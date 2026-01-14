# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
from transformers import BertTokenizer
import loader

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model_path"])
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        # vocab_size = len(self.tokenizer.vocab)
        vocab_size = 21128

        # self.dropout = nn.Dropout(0.1)
        self.classify = nn.Linear(self.bert.config.hidden_size, vocab_size)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # ignore_index=-1 是表示忽略值为-1(填充位)的元素，使其不参与交叉熵的计算

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask=None, target=None):
        # 调用 bert， [batch_size, sen_len] -> [batch_size, sen_len, embed_size]
        sequence_out, _ = self.bert(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=False)
        # [batch_size, sen_len, embed_size] -> [batch_size, sen_len, vocab_size]
        logits = self.classify(sequence_out)

        if target is not None:
            # [batch_size * sen_len, vocab_size] , [batch_size * sen_len]
            return self.loss(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        else:
            # [batch_size, sen_len, vocab_size]
            return logits


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


# def dispach_gpu(model):
#     if torch.cuda.device_count() > 1:
#         print(f"使用 {torch.cuda.device_count()} 个GPU")
#         model = nn.DataParallel(model)  # 这会自动将模型放到所有可用GPU上
#     return model.cuda()


if __name__ == "__main__":
    from config import Config

    local_data = loader.load_train_data(Config["valid_data_path"], Config)

    model = TorchModel(Config)
    for index, batch_data in enumerate(local_data):
        inputs_ids, attention_mask, target_ids = batch_data
        print("inputs_ids: ", inputs_ids)
        print("target_ids: ", target_ids)
        print("attention_mask: ", attention_mask)
        loss = model.forward(inputs_ids, attention_mask, target_ids)
        print("loss: ", loss.item())
        break
