# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

"""
数据加载
"""


class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])  # 标准问的label及id: { "停机保号": 0,"密码重置": 1, ...}
        self.epoch_sample_size = config["epoch_sample_size"]  # 随机采样的训练样本数量
        self.data_type = None  # 用来标识加载的是训练集还是测试集 "train" or "test"
        self._load_data()

    def _load_data(self):
        self.data = []
        self.knwb = defaultdict(list)  # {标准问id: [常用问1的向量, 常用问2的向量], ...}
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        std_ques_id = self.schema[label]  # 标准问id
                        ques_idvec = self.text_to_idvec(question)
                        self.knwb[std_ques_id].append(torch.LongTensor(ques_idvec))  # {标准问id: [常用问1的向量, 常用问2的向量], ...}
                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    std_ques_id = self.schema[label]  # 标准问id
                    ques_idvec = self.text_to_idvec(question)
                    self.data.append(
                        [torch.LongTensor(ques_idvec), torch.LongTensor([std_ques_id])])  # {常用问的向量:[标准问id])}
        return

    def text_to_idvec(self, text):
        vec = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                vec.append(self.vocab.get(word, self.vocab["<unk>"]))
        else:
            for char in text:
                vec.append(self.vocab.get(char, self.vocab["<unk>"]))

        vec = vec[:self.config["sentence_len"]]  # 截断
        vec += [0] * (self.config["sentence_len"] - len(vec))  # 补齐
        return vec

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_sample_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    # 随机选择两个标准问，从其中一个标准问中选择两个常用问，再从另一个标准问选择一个常用句，这三个常用问构成一份样本。
    def random_train_sample(self):
        std_question_ids = list(self.knwb.keys())
        std_ques_id1, std_ques_id2 = random.sample(std_question_ids, 2)  # 随机选取两个标准问

        # 如果两个标准问下都不足两个问题，则重新随机一次
        if len(self.knwb[std_ques_id1]) < 2 and len(self.knwb[std_ques_id1]) < 2:
            return self.random_train_sample()

        if len(self.knwb[std_ques_id1]) >= 2:
            a_vec1, p_vec1 = random.sample(self.knwb[std_ques_id1], 2)  # 从标准问中选择两个常用问
            n_vec1 = random.choice(self.knwb[std_ques_id2])  # 从标准问中选择两个常用问
            return [a_vec1, p_vec1, n_vec1]
        else:
            a_vec1, p_vec1 = random.sample(self.knwb[std_ques_id2], 2)  # 从标准问中选择两个常用问
            n_vec1 = random.choice(self.knwb[std_ques_id1])  # 从标准问中选择两个常用问
            return [a_vec1, p_vec1, n_vec1]


# 加载字表或词表
def load_vocab(vocab_path):
    char_dict = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            char_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    char_dict["<unk>"] = len(char_dict)
    return char_dict


# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


# 用torch自带的DataLoader类封装数据
def load_data(path, config, shuffle=True):
    dg = DataGenerator(path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("../data/train.json", Config)
    print(dg[1])

    # dg = DataGenerator("../data/test.json", Config)
    # print(dg[1])
