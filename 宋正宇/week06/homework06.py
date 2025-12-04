# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json


class TorchModel(nn.Module):
    def __init__(self, sentence_lenth, hidden_size, num_attention_heads, num_classes):
        super(TorchModel, self).__init__()
        self.token_embedding = nn.Embedding(sentence_lenth, hidden_size, padding_idx=0)  # embedding层
        self.segment_embedding = nn.Embedding(2, hidden_size, padding_idx=0)  # embedding层
        self.position_embedding = nn.Embedding(512, hidden_size, padding_idx=0)  # embedding层

        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)

        self.activation = nn.Softmax(dim=1)

        self.layer_norm1 = nn.LayerNorm(512)

        self.feed1 = nn.Linear(4*hidden_size, 4*hidden_size)
        self.feed2 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.layer_norm2 = nn.LayerNorm(512)


    def forward(self, x):
        XQ = self.token_embedding(x)
        XK = self.segment_embedding(x)
        XV = self.position_embedding(x)
        x_encoding = XQ + XK + XV
        WQ = self.W_Q(x_encoding)
        WK = self.W_K(x_encoding)
        WV = self.W_V(x_encoding)
        Q = x_encoding * WQ
        K = x_encoding * WK
        V = x_encoding * WV
        QKV = self.activation((Q * K.T) / 8) * V
        QKV = QKV.swapaxes(0, 1).reshape(-1, self.hidden_size)
        x_decoding = self.layer_norm1(x_encoding + QKV)
        x_feed1 = self.feed1(x_decoding)
        x_gelu = self.gelu(x_feed1)
        x_feed2 = self.feed2(x_gelu)
        x_output = self.layer_norm2(x_decoding + x_feed2)
        return x_output







