# -*- coding:utf-8 -*-

import torch
from config import Labels
from torch.utils.data import Dataset, DataLoader

class SentenceLabelLoader(Dataset):
    def __init__(self, config, data_path):
        super(SentenceLabelLoader, self).__init__()
        self.data_path = data_path
        self.vocab_dict = load_vocab(config["vocab_path"])
        self.max_length = config["max_length"]
        self.data = []
        self.load()

    def load(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = ""
                label = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    c, l = line.strip().split(" ")
                    sentence = f"{sentence}{c}"
                    label.append(Labels[l])
                sequence = sentence2sequence(sentence, self.vocab_dict, self.max_length)
                label = padding(label, self.max_length, pad_token=-1)
                # 如果样本中没有实体，则丢弃（净化训练数据）
                if all(x == 0 or x == -1 for x in label):
                    continue
                self.data.append([torch.LongTensor(sequence), torch.LongTensor(label)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 补齐或截断（仅对列表或字符串进行操作）
def padding(seq, max_length, pad_token=0):
    assert isinstance(seq, (list, str))
    seq = seq[:max_length]
    if len(seq) < max_length:
        seq += [pad_token] * (max_length - len(seq))
    return seq

# 将文本转化为序列（附带补齐和截断）
def sentence2sequence(sentence, vocab, max_length):
    sequence = []
    unk_idx = vocab["[UNK]"]
    pad_idx = vocab["[PAD]"]
    sentence = sentence[: max_length]
    for char in sentence:
        idx = vocab.get(char, unk_idx)
        sequence.append(idx)
    if len(sentence) < max_length:
        count = max_length - len(sentence)
        sequence += [pad_idx] * count
    return sequence


def load_data(config, data_path):
    dataset = SentenceLabelLoader(config, data_path)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

def load_vocab(vocab_path):
    vocab_dict = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for index, char in enumerate(f):
            vocab_dict[char.strip()] = index
    return vocab_dict