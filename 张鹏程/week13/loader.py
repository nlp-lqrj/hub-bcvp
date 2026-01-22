import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载 - NER任务
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {"B-LOCATION": 0,
                               "B-ORGANIZATION": 1,
                               "B-PERSON": 2,
                               "B-TIME": 3,
                               "I-LOCATION": 4,
                               "I-ORGANIZATION": 5,
                               "I-PERSON": 6,
                               "I-TIME": 7,
                               "O": 8
                               }
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        """加载NER数据，格式为每行一个token和标签，空行分隔句子"""
        self.data = []
        self.sentences = []  # 保存原始句子（字符列表）
        tokens = []
        labels = []

        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:  # 空行表示句子结束
                    if tokens:
                        self.process_sentence(tokens, labels)
                        tokens = []
                        labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[1]
                        tokens.append(token)
                        labels.append(label)

            # 处理最后一个句子（如果文件末尾没有空行）
            if tokens:
                self.process_sentence(tokens, labels)
        return

    def process_sentence(self, tokens, labels):
        """处理一个句子，将tokens和labels编码为模型输入格式"""
        # 保存原始句子（字符列表）和对应的原始标签
        self.sentences.append(tokens.copy())
        # 将tokens组合成文本
        text = "".join(tokens)

        # 使用tokenizer编码
        if self.config["model_type"] == "bert":
            # 使用encode_plus获取更多信息
            encoded = self.tokenizer.encode_plus(
                text,
                max_length=self.config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=False
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            # 对齐labels：对于中文BERT，需要将字符级别的标签对齐到token级别
            # 关键问题：逐个字符tokenize和整个句子tokenize的结果可能不一致
            # 解决方法：使用整个句子的tokenize结果，然后通过字符位置对齐
            aligned_labels = []

            # 跳过[CLS]
            aligned_labels.append(-100)

            # 使用整个句子的tokenize结果来对齐
            # 对于中文BERT，通常一个字符对应一个token（或少数几个subword token）
            # 我们需要找到每个token对应的字符位置
            char_pos = 0  # 当前字符位置

            # 逐个字符处理，找到对应的token
            for char_idx, char in enumerate(tokens):
                if char_idx >= len(labels):
                    break

                # tokenize当前字符（用于确定需要多少个token）
                char_tokens = self.tokenizer.encode(char, add_special_tokens=False)
                label_str = labels[char_idx]
                label_id = self.label_to_index.get(label_str, 8)

                # 第一个subword token使用原label
                if len(char_tokens) > 0:
                    aligned_labels.append(label_id)
                    # 如果有多个subword token，其余设为-100
                    for _ in range(len(char_tokens) - 1):
                        aligned_labels.append(-100)

                # 检查是否超过最大长度（-1 for [SEP]）
                if len(aligned_labels) >= self.config["max_length"] - 1:
                    break

            # 添加[SEP]和padding
            aligned_labels.append(-100)  # [SEP]
            while len(aligned_labels) < self.config["max_length"]:
                aligned_labels.append(-100)  # padding

            # 确保长度一致
            aligned_labels = aligned_labels[:self.config["max_length"]]

            labels_tensor = torch.LongTensor(aligned_labels)
        else:
            # 非BERT模型的处理
            input_ids = self.encode_sentence(text)
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.ones_like(input_ids)

            # 对齐labels
            aligned_labels = []
            for i, label_str in enumerate(labels):
                aligned_labels.append(self.label_to_index.get(label_str, 8))
            # padding labels
            while len(aligned_labels) < self.config["max_length"]:
                aligned_labels.append(8)  # O
            aligned_labels = aligned_labels[:self.config["max_length"]]
            labels_tensor = torch.LongTensor(aligned_labels)

        self.data.append([input_ids, attention_mask, labels_tensor])

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
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
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])