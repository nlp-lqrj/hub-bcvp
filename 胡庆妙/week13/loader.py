# -*- coding: utf-8 -*-

import json
import torch
import jieba
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator(Dataset):
    def __init__(self, path, config):
        self.path = path
        self.config = config
        self.schema = load_schema(config["schema_path"])  # {标注的label: 类别}, 9类标注映射为9个类别
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])

        self.sentences = []  # 从语料转换得来的自然语句
        self.data = []
        self.load()

    def load(self):
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:  # 每段对应一个句子
                sentence = []
                label_ids = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)  # 语句中的字
                    label_ids.append(self.schema[label])  # 语句中字的标注 -> 类别, 9类标注映射为9个类别
                sentence = "".join(sentence)
                self.sentences.append(sentence)

                input_ids = encode_sentence(self.tokenizer, sentence, max_length=self.config["sentence_len"])
                label_ids = truncate_padding(label_ids, self.config["sentence_len"], pad_token=-1)
                # [一句自然语句的字序列, 一句自然语句的标注序列]
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(label_ids)])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def encode_sentence(tokenizer, sentence, max_length=0, truncate=True, padding=True):
    input_ids = []
    for char in sentence:
        input_ids.append(tokenizer.vocab.get(char, tokenizer.vocab["[UNK]"]))

    return truncate_padding(input_ids, max_length, pad_token=tokenizer.vocab["[PAD]"], truncate=truncate, padding=padding)


def truncate_padding(seq_list, max_length, pad_token=0, truncate=True, padding=True):
    if truncate:
        seq_list = seq_list[:max_length]
    if padding:
        seq_list += [pad_token] * (max_length - len(seq_list))
    return seq_list


def load_schema(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


# 用torch自带的DataLoader类封装数据
def load_data(path, config, shuffle=True):
    dg = DataGenerator(path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    local_data = load_data("data/train.txt", Config)

    for index, batch_data in enumerate(local_data):
        input_ids, labels = batch_data
        print(">>", input_ids, labels)
