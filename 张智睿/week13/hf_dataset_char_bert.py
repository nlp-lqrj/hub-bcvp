# -*- coding: utf-8 -*-
"""用于读取 ner_data/* 字符级 NER 数据的 HuggingFace Dataset。

输入数据格式：
  - 样本之间用空行分隔
  - 每个非空行：<字符> <标签>
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from torch.utils.data import Dataset


def read_conll_char(path: str) -> List[Tuple[List[str], List[str]]]:
    samples: List[Tuple[List[str], List[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return samples
    blocks = content.split("\n\n")
    for b in blocks:
        chars: List[str] = []
        tags: List[str] = []
        for line in b.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                # 保持容错：遇到格式异常的行直接跳过
                continue
            c, t = parts
            chars.append(c)
            tags.append(t)
        if chars:
            samples.append((chars, tags))
    return samples


class CharBertNerDataset(Dataset):
    """Character-as-word dataset for BERT token classification.

    We feed `chars` as a list with `is_split_into_words=True`. For Chinese,
    this usually yields a 1-to-1 mapping (char -> token). For any multi-subtoken
    wordpiece case, we label only the first subtoken and set the rest to -100.
    """

    def __init__(
        self,
        data_path: str,
        schema_path: str,
        tokenizer,
        max_length: int = 128,
    ):
        self.samples = read_conll_char(data_path)
        with open(schema_path, "r", encoding="utf-8") as f:
            self.label2id: Dict[str, int] = json.load(f)
        self.id2label: Dict[int, str] = {v: k for k, v in self.label2id.items()}
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

        # 为旧版“正则解码”评测口径保留每条样本的原始句子文本
        self.sentences: List[str] = ["".join(chars) for chars, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chars, tags = self.samples[idx]

        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )

        word_ids = encoding.word_ids()
        labels: List[int] = []
        prev_wid: Optional[int] = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)  # 特殊符号位置（如起始符/分隔符）忽略
            elif wid != prev_wid:
                # 该字符对应的第一个子词位置才保留标签
                tag = tags[wid]
                labels.append(self.label2id[tag])
            else:
                labels.append(-100)  # 该字符的后续子词位置忽略
            prev_wid = wid

        encoding["labels"] = labels
        return encoding
