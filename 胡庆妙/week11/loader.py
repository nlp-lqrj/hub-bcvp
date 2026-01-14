# -*- coding: utf-8 -*-

import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator(Dataset):
    def __init__(self, path, config, data_type="train"):
        self.path = path
        self.config = config

        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.tokenizer.vocab[" "] = len(self.tokenizer.vocab)  # 在词表中增加空格符

        self.data_type = data_type
        self.data = []
        self.load()

    def load(self):
        with open(self.path, encoding="utf8") as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                ask = line["title"].strip()  # 样本中的问句
                ans = line["content"].strip()  # 样本中的答句

                if self.data_type == "train":
                    train_data = self.build_train_sample(ask, ans, self.config["max_length"])
                    input_ids, attention_mask, target_ids = train_data
                    self.data.append([input_ids, attention_mask, target_ids])
                else:
                    self.data.append([ask, ans])
        return

    def build_train_sample(self, ask, ans, max_len):
        # 问句序列、答句序列
        ask_seq = self.tokenizer.encode(ask, add_special_tokens=False)
        ans_seq = self.tokenizer.encode(ans, add_special_tokens=False)
        # [CLS] + ask_seq + [SEP] + ans_seq
        input_ids = ([self.tokenizer.cls_token_id] + ask_seq
                     + [self.tokenizer.sep_token_id]  # 中间[SEP]是分隔符
                     + ans_seq)
        # ask_seq + [SEP] + ans_seq + [SEP], 且问题部分设为-1(不参与loss计算)
        target_ids = ([-1] + len(ask_seq) * [-1]
                      + ans_seq
                      + [self.tokenizer.sep_token_id])  # 末尾[SEP]标识语句结束
        # 交叉注意力掩码
        encoder_input_mask = np.ones((len(ask_seq) + 2, len(ask_seq) + 2), dtype=np.int32)  # 左上，全1
        decoder_input_mask = np.zeros((len(ask_seq) + 2, len(ans_seq)), dtype=np.int32)  # 右上，全0
        encoder_ouput_mask = np.ones((len(ans_seq), len(ask_seq) + 2), dtype=np.int32)  # 左下，全1
        decoder_output_mask = np.tril(np.ones((len(ans_seq), len(ans_seq)), dtype=np.int32))  # 右下，下三角
        input_mask = np.hstack([encoder_input_mask, decoder_input_mask])  # 水平拼接
        output_mask = np.hstack([encoder_ouput_mask, decoder_output_mask])  # 水平拼接
        attention_mask = np.vstack([input_mask, output_mask])  # 垂直拼接

        # 截长、补齐
        input_ids = input_ids[:max_len] + [self.tokenizer.pad_token_id] * (max_len - len(input_ids))
        target_ids = target_ids[:max_len] + [self.tokenizer.pad_token_id] * (max_len - len(target_ids))
        if attention_mask.shape[0] > max_len:
            from_idx = min(len(input_ids), max_len)
            attention_mask = attention_mask[:from_idx, :from_idx]
        elif attention_mask.shape[0] < max_len:
            pad_size = max_len - attention_mask.shape[0]
            pad_width = ((0, pad_size), (0, pad_size))  # 行下方加pad_size行，列右侧加pad_size列
            attention_mask = np.pad(attention_mask, pad_width, mode='constant', constant_values=0)

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(target_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 用torch自带的DataLoader类封装数据
def load_train_data(data_path, config):
    dg = DataGenerator(data_path, config, "train")
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=True)
    return dl


# 用torch自带的DataLoader类封装数据
def load_valid_data(data_path, config, batch_size=1):
    dg = DataGenerator(data_path, config, "valid")
    dl = DataLoader(dg, batch_size=batch_size, shuffle=False)
    return dl


if __name__ == "__main__":
    from config import Config

    l_train_data = load_train_data(Config["train_data_path"], Config)
    for index, batch_data in enumerate(l_train_data):
        inpt_ids, atten_mask, trg_ids = batch_data
        print("inputs_ids:", inpt_ids)
        print("attention_mask:", atten_mask)
        print("target_ids:", trg_ids)
        break

    l_valid_data = load_valid_data(Config["valid_data_path"], Config, batch_size=2)
    for index, batch_data in enumerate(l_valid_data):
        ask, ans = batch_data
        print("ask:", ask[0])
        print("ans:", ans[0])
        break
