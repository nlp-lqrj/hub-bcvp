#coding:utf8

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
import math
import random
from transformers import BertModel, BertTokenizer
from transformers import BertConfig
"""
基于pytorch的BERT语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, vocab, hidden_size=768, pretrained_model_name=r"E:\newlife\badou\第六周 语言模型\bert-base-chinese"):
        super(LanguageModel, self).__init__()
        # 修改BERT配置，只保留3层
        config = BertConfig.from_pretrained(pretrained_model_name)
        config.num_hidden_layers = 2  # 只保留2层
        # config.vocab_size = len(vocab)
        self.bert = BertModel.from_pretrained(
            pretrained_model_name,
            config=config  # 使用修改后的配置
        )
        # 分类层
        self.classify = nn.Linear(hidden_size, len(vocab))
        # self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        attention_mask = (x != 0).long()  # 创建attention mask，非padding位置为1

        # BERT前向传播
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        # 只取最后一个 token 的输出（即输入的最后一个字的表示）
        last_hidden = outputs[0][:, -1, :]  # [batch, hidden_size]
        y_pred = self.classify(last_hidden)  # [batch, vocab_size]
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    # 输入: window_size 个字
    # 输出: 只预测第 window_size+1 个字（单个）
    start = random.randint(0, len(corpus) - 1 - window_size)
    x_window = corpus[start:start + window_size]          # [w0, w1, ..., w9]
    y_char = corpus[start + window_size]                  # w10 （只预测一个！）
    # print(window, target)
    x = [vocab.get(ch, vocab["<UNK>"]) for ch in x_window]
    y = vocab.get(y_char, vocab["<UNK>"])
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab):
    model = LanguageModel(vocab)
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = {idx: char for char, idx in vocab.items()}
    model.eval()
    with torch.no_grad():
        while len(openings) <= 30:
            # 取最后 window_size 个字符
            recent = openings[-window_size:]
            # 转为 token IDs
            x = [vocab.get(ch, vocab["<UNK>"]) for ch in recent]
            # 如果长度不足 window_size，前面补 <pad> 的 ID（即 0）
            if len(x) < window_size:
                x = [vocab["<pad>"]] * (window_size - len(x)) + x
            x = torch.LongTensor([x]).to(next(model.parameters()).device)
            prob = model(x)  # [1, vocab_size]
            index = sampling_strategy(prob[0])
            pred_char = reverse_vocab.get(index, "<UNK>")
            if pred_char == "<UNK>" or pred_char == "<pad>":
                break
            openings += pred_char
            if pred_char in "。！？\n":
                break
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab.get("<UNK>", 0)) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab.get("<UNK>", 0))
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 32       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    window_size = 10       #样本文本长度
    vocab = build_vocab("vocab.txt")       #建立字表
    vocab["<UNK>"] = len(vocab)  # 添加UNK token
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab)    #建立模型

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")

    print("开始构建数据集...")
    # 一次性构建所有训练样本，避免在每个epoch和batch重复生成
    full_dataset_x, full_dataset_y = build_dataset(train_sample, vocab, window_size, corpus)
    print(f"数据集构建完毕，样本形状: x={full_dataset_x.shape}, y={full_dataset_y.shape}")

    # ==================== 优化1: 设置CPU并行计算
    # 设置PyTorch使用多核CPU并行计算
    # 查看实际物理核心数（非超线程）
    import multiprocessing
    torch.set_num_threads(multiprocessing.cpu_count() // 2)  # 避免超线程争抢

    # 将模型设置为评估模式以加速（训练时会自动切换）
    model.eval()

    # 预热：提前编译一些计算图
    print("正在进行预热编译...")
    with torch.no_grad():
        _ = model(full_dataset_x[:batch_size], full_dataset_y[:batch_size])
    model.train()
    print("预热完成")
    # ===============================================================

    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)   #建立优化器

    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # 改为从预加载的数据集中按批次读取
        for batch in range(int(train_sample / batch_size)):
            start = batch * batch_size
            end = start + batch_size
            x = full_dataset_x[start:end]
            y = full_dataset_y[start:end]
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss

            # 在CPU上，梯度裁剪可以稳定训练，避免数值问题
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())

        # 每5轮降低学习率
        if (epoch + 1) % 5 == 0:
            for param_group in optim.param_groups:
                param_group['lr'] *= 0.5
            print(f"学习率降至: {optim.param_groups[0]['lr']:.6f}")

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
