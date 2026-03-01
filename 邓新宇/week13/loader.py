# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast  # 导入BERT快速分词器，支持offset_mapping

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        
        # BERT模型相关初始化
        self.use_bert = config["use_bert"]  # 检查是否使用BERT
        if self.use_bert:
            # 初始化BERT快速分词器
            self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])
        else:
            # 原有字表加载方式
            self.vocab = load_vocab(config["vocab_path"])
            self.config["vocab_size"] = len(self.vocab)
        
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))
                
                if self.use_bert:
                    # 对于BERT模型，需要处理子词分割的情况
                    text = "".join(sentenece)
                    # 先获取原始字符级别的tokenization
                    encoded = self.tokenizer(text, padding=False, truncation=False, add_special_tokens=False, return_offsets_mapping=True)
                    input_ids = encoded["input_ids"]
                    offset_mapping = encoded["offset_mapping"]
                    
                    # 扩展标签以匹配子词序列长度
                    extended_labels = []
                    char_idx = 0
                    for token_offset in offset_mapping:
                        if token_offset == (0, 0):
                            # 跳过特殊标记（虽然我们设置了add_special_tokens=False，但还是做一下安全检查）
                            extended_labels.append(-1)
                        else:
                            # 对于子词，使用对应字符的标签
                            extended_labels.append(labels[char_idx])
                            # 如果token结束位置大于当前字符结束位置，说明是多字节字符或子词
                            if token_offset[1] > (char_idx + 1):
                                char_idx += 1
                    
                    # 对输入和标签进行padding
                    input_ids = self.padding(input_ids)
                    extended_labels = self.padding(extended_labels, -1)
                    
                    self.data.append([torch.LongTensor(input_ids), torch.LongTensor(extended_labels)])
                else:
                    # 原有处理方式
                    input_ids = self.encode_sentence(sentenece)
                    labels = self.padding(labels, -1)
                    self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        if self.use_bert:
            # 使用BERT分词器进行tokenization
            encoded = self.tokenizer(text, padding=False, truncation=False, add_special_tokens=False)
            input_id = encoded["input_ids"]
        else:
            # 原有编码方式
            input_id = []
            if self.config["vocab_path"] == "words.txt":
                for word in jieba.cut(text):
                    input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
            else:
                for char in text:
                    input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
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
    dg = DataGenerator("../ner_data/train.txt", Config)

