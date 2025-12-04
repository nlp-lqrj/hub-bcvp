import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from transformers import BertModel
from triton.language import dtype


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, chr_dim = 512):
        super(BertEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, chr_dim, padding_idx=0)
        self.seg_embed = nn.Embedding(2, chr_dim)
        self.pos_embed = nn.Embedding(512, chr_dim)
        self.norm = nn.LayerNorm(chr_dim)
    def forward(self, x):
        # x  = [batch, seq_len]
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        seg = torch.ones_like(x, dtype=torch.long)
        x = self.embed(x) + self.seg_embed(seg) + self.pos_embed(pos)
        return self.norm(x)
class MultiHeadAttention(nn.Module):
    def __init__(self, chr_dim = 512, num_heads = 8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.chr_dim = chr_dim
        self.d_k = chr_dim // num_heads
        self.W_Q = nn.Linear(chr_dim, chr_dim)
        self.W_K = nn.Linear(chr_dim, chr_dim)
        self.W_V = nn.Linear(chr_dim, chr_dim)
        self.W_out = nn.Linear(chr_dim, chr_dim)
        self.norm = nn.LayerNorm(chr_dim)
    def forward(self, q, k, v, mask=None):
        q_s = self.W_Q(q).view(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(k).view(k.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(v).view(v.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q_s, k_s.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.matmul(attn, v_s)
        output = self.W_out(attn.transpose(1, 2).contiguous().view(q.size(0), -1, self.num_heads * self.d_k))
        return output



class DiyBertModel(nn.Module):
    def __init__(self, vocab_size, chr_dim = 512):
        super(DiyBertModel, self).__init__()
        self.embed = BertEmbedding(vocab_size, chr_dim)
        self.attn = MultiHeadAttention(chr_dim = chr_dim)
        self.norm1 = nn.LayerNorm(chr_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(chr_dim, chr_dim * 4),
            nn.GELU(),
            nn.Linear(chr_dim * 4, chr_dim),
        )
        self.norm2 = nn.LayerNorm(chr_dim)
    def forward(self, x):
        x = self.embed(x)
        x = self.norm1(x + self.attn(x, x, x))
        x = self.norm2(x + self.feed_forward(x))
        return x



if __name__ == "__main__":
  
    bert = BertModel.from_pretrained(r"/mnt/2T/wwk/pycharm_environment/study/google-bert/bert-base-chinese", return_dict=False)
    state_dict = bert.state_dict()
    bert.eval()
    x = np.array([2450, 15486, 102, 2110])   #假想成4个字的句子
    torch_x = torch.LongTensor([x])          #pytorch形式输入
    print(torch_x.shape)
    seqence_output, pooler_output = bert(torch_x)
    print(seqence_output.shape, pooler_output.shape)
    # print(seqence_output, pooler_output)

    x1 = np.array([[2450, 15486, 102, 2110],
                  [2450, 15486, 102, 2110],
                   [2450, 15486, 102, 2110],
                   ])   #假想成4个字的句子，总共有 3 句
    torch_x1 = torch.LongTensor(x1)          #pytorch形式输入
    print(torch_x1.shape)
    seqence_output, pooler_output = bert(torch_x1)
    print(seqence_output.shape, pooler_output.shape)
    print('=' * 20)
    mod = DiyBertModel(vocab_size = 100000, chr_dim = 512)
    print(f'自己diy的bert模型的输出{mod(torch_x1)}，形状大小为：{mod(torch_x1).shape}')
    # print(seqence_output, pooler_output)
    print('=' * 20)
    sums = 0
    for i, k in state_dict.items():
        j = 1
        for a in k.shape:
            j *= a
        sums += j
        print(f'参数{i}是{k.shape}')
    print('=' * 30)
    print(f'Bert中的总参数大小为{sums}个float32值')


