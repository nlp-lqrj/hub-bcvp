# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
            # self.vocab = self.tokenizer.get_vocab
            self.config["vocab_path"] = self.config["pretrain_model_path"] + "\\vocab.txt"
            self.config["vocab_size"] = self.tokenizer.vocab_size
        else:
            self.vocab = load_vocab(config["vocab_path"])
            self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                # tag = line["tag"]
                # label = self.label_to_index[tag]
                label = line["label"]
                review = line["review"]
                if self.config["model_type"] == "bert":
                    # 使用tokenizer（分词器）对文本进行编码处理。这行代码将文本(title)转换为数字序列
                    # max_length: 设置序列的最大长度，来自配置文件
                    # pad_to_max_length=True: 表示如果文本长度小于max_length，会自动填充到指定长度

                    # Keyword arguments {'pad_to_max_length': True} not recognized. 表明你使用的tokenizer版本中不支持pad_to_max_length参数。
                    # input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                    input_id = self.tokenizer.encode(review, max_length=self.config["max_length"],
                                                     padding='max_length', truncation=True)  # 这里返回的是一个list

                    # padding='max_length'：将序列填充到max_length指定的长度
                    # truncation=True：如果序列超过max_length，则进行截断
                else:
                    input_id = self.encode_sentence(review)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]  # 这行代码对序列进行截断操作，如果序列长度超过self.max_length，则保留前self.max_length个元素，如果序列长度小于等于self.max_length，则保持原样
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    # dg = DataGenerator("../data/valid_tag_news.json", Config)
    dg = DataGenerator("../data/valid_reviews.json", Config)
    print(dg[1])
