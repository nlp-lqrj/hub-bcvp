#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertModel, BertTokenizer

"""
基于BERT的自回归语言模型
"""

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, bert_model_path="E:\\BaiduNetdiskDownload\\第六周 语言模型\\bert-base-chinese"):
        super(LanguageModel, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(bert_model_path)
        
        # 获取BERT的隐藏层大小
        self.hidden_size = self.bert.config.hidden_size
        
        # 分类层：将BERT的输出映射到词汇表大小
        self.classify = nn.Linear(self.hidden_size, vocab_size)
        
        # Dropout层
        self.dropout = nn.Dropout(0.1)
        
        # 损失函数
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None, attention_mask=None):
        # 通过BERT模型
        outputs = self.bert(x, attention_mask=attention_mask)
        
        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        else:
            sequence_output = outputs.last_hidden_state
        
        # 通过分类层得到预测结果
        y_pred = self.classify(sequence_output)  # shape: (batch_size, seq_len, vocab_size)
        
        if y is not None:
            # 计算损失
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 返回概率分布
            return torch.softmax(y_pred, dim=-1)

# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0, "<unk>": 1, "<mask>": 2}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 3  # 留出0位给pad token，1位给unk，2位给mask
    return vocab

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    # print(window, target)
    x = [vocab.get(word, vocab["<unk>"]) for word in window]  # 将字转换成序号
    y = [vocab.get(word, vocab["<unk>"]) for word in target]
    return x, y

# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, bert_model_path="E:\\BaiduNetdiskDownload\\第六周 语言模型\\bert-base-chinese"):
    model = LanguageModel(len(vocab), bert_model_path)
    return model

# 创建因果掩码（自回归掩码）
def create_causal_mask(batch_size, seq_len, device):
    """
    创建因果掩码，确保模型只能看到前面的词
    """
    # 创建下三角矩阵，对角线及以下为True，其余为False
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).expand(batch_size, -1, -1)
    # BERT需要的是float类型的mask，且1表示可用，0表示不可用
    mask = mask.float()
    return mask

# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            # 取窗口大小的输入
            input_text = openings[-window_size:]
            x = [vocab.get(char, vocab["<unk>"]) for char in input_text]
            
            # 确保输入长度为window_size，不足的用pad填充
            if len(x) < window_size:
                x = [vocab["<pad>"]] * (window_size - len(x)) + x
            
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            
            # 创建因果掩码
            causal_mask = create_causal_mask(1, x.size(1), x.device)
            
            # 获取模型输出
            outputs = model.bert(x, attention_mask=causal_mask)
            if isinstance(outputs, tuple):
                sequence_output = outputs[0]
            else:
                sequence_output = outputs.last_hidden_state
            y_pred = model.classify(sequence_output)
            
            # 取最后一个位置的预测结果
            y = y_pred[0][-1]
            # 应用softmax确保概率分布非负
            y = torch.softmax(y, dim=-1)
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

def sampling_strategy(prob_distribution):
    # 确保概率分布中的值非负
    prob_distribution = torch.clamp(prob_distribution, min=0.0)
    # 确保概率和为1
    prob_distribution = prob_distribution / prob_distribution.sum()
    
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<unk>"]) for char in window]
            
            # 确保输入长度为window_size，不足的用pad填充
            if len(x) < window_size:
                x = [vocab["<pad>"]] * (window_size - len(x)) + x
            
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<unk>"])
            if torch.cuda.is_available():
                x = x.cuda()
            
            # 创建因果掩码
            causal_mask = create_causal_mask(1, x.size(1), x.device)
            
            # 获取模型输出
            outputs = model.bert(x, attention_mask=causal_mask)
            if isinstance(outputs, tuple):
                sequence_output = outputs[0]
            else:
                sequence_output = outputs.last_hidden_state
            y_pred = model.classify(sequence_output)
            
            # 取最后一个位置的预测概率
            pred_prob_distribute = torch.softmax(y_pred[0][-1], dim=-1)  # 应用softmax
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))

def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 16  # 每次训练样本个数，由于BERT较大，减小batch_size以节省内存
    train_sample = 10000  # 每轮训练总共训练的样本总数，减少以加快训练
    window_size = 10  # 样本文本长度
    vocab = build_vocab("vocab.txt")  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)  # BERT通常使用较小的学习率
    print("BERT语言模型加载完毕，开始训练")
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            
            # 创建因果掩码
            causal_mask = create_causal_mask(batch_size, x.size(1), x.device)
            
            optim.zero_grad()  # 梯度归零
            loss = model(x, y, attention_mask=causal_mask)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
