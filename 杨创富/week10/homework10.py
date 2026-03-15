


#coding:utf8
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertConfig, BertModel, BertTokenizer

"""
基于BERT的自回归语言模型（带因果掩码）
"""

class BertAutoRegressiveModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=6, num_heads=8):
        super(BertAutoRegressiveModel, self).__init__()
        
        # 使用BERT配置
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=512,
            is_decoder=True,  # 设置为解码器
            add_cross_attention=False,
            is_encoder_decoder=False,
        )
        
        self.bert = BertModel(config)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def create_causal_mask(self, seq_len, device):
        """创建因果掩码（防止看到未来信息）"""
        mask = torch.ones(seq_len, seq_len, device=device).tril(diagonal=0)
        return mask.bool()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 创建因果注意力掩码
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 生成因果掩码
        causal_mask = self.create_causal_mask(seq_len, device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
        
        # BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=causal_mask.squeeze(1)  # 使用因果掩码
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        logits = self.classifier(sequence_output)    # [batch_size, seq_len, vocab_size]
        
        if labels is not None:
            # 计算损失（忽略padding）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return loss
        else:
            return torch.softmax(logits, dim=-1)

# 加载BERT词表（使用预训练的词表或从头训练）
def load_bert_vocab(vocab_path):
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer

# 构建BERT格式的样本
def build_bert_sample(tokenizer, window_size, corpus):
    # 对文本进行分词
    tokens = tokenizer.encode(corpus, add_special_tokens=False)
    
    if len(tokens) < window_size + 1:
        # 补全或处理短文本
        return None, None
    
    start = random.randint(0, len(tokens) - window_size - 1)
    input_ids = tokens[start:start + window_size]
    target_ids = tokens[start + 1:start + window_size + 1]
    
    return input_ids, target_ids

# 构建数据集
def build_bert_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    
    while len(dataset_x) < sample_length:
        x, y = build_bert_sample(tokenizer, window_size, corpus)
        if x is not None and y is not None:
            dataset_x.append(x)
            dataset_y.append(y)
    
    # 添加特殊token并padding
    max_len = window_size
    padded_x = []
    padded_y = []
    
    for x, y in zip(dataset_x, dataset_y):
        # 添加[CLS]和[SEP]标记（可选）
        padded_x.append(x[:max_len] + [0] * (max_len - len(x)))
        padded_y.append(y[:max_len] + [0] * (max_len - len(y)))
    
    return torch.LongTensor(padded_x), torch.LongTensor(padded_y)

# 文本生成函数
def generate_text_with_bert(model, tokenizer, prompt, max_length=50, temperature=1.0):
    model.eval()
    generated = tokenizer.encode(prompt, add_special_tokens=False)
    
    with torch.no_grad():
        for _ in range(max_length):
            # 准备输入
            input_ids = torch.LongTensor([generated[-window_size:]]).to(next(model.parameters()).device)
            
            # 获取预测
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # 采样下一个token
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1).item()
            
            generated.append(next_token)
            
            # 如果生成结束标记则停止
            if next_token == tokenizer.sep_token_id:
                break
    
    # 解码生成的文本
    return tokenizer.decode(generated, skip_special_tokens=True)

# 计算困惑度
def calculate_bert_perplexity(model, tokenizer, text):
    model.eval()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    total_log_prob = 0
    with torch.no_grad():
        for i in range(1, len(tokens)):
            input_ids = torch.LongTensor([tokens[max(0, i-window_size):i]]).to(next(model.parameters()).device)
            outputs = model(input_ids)
            
            # 获取最后一个token的预测分布
            probs = outputs[0, -1, :]
            target_prob = probs[tokens[i]].item()
            
            if target_prob > 0:
                total_log_prob += math.log(target_prob)
    
    return math.exp(-total_log_prob / (len(tokens) - 1))

# 训练函数
def train_bert_model(corpus_path, vocab_path, save_path="bert_lm.pth"):
    # 超参数
    epoch_num = 10
    batch_size = 16  # BERT较大，批大小较小
    train_sample = 20000
    window_size = 64  # 可以设置更长
    hidden_size = 256  # 可以调整
    
    # 加载词表和语料
    tokenizer = load_bert_vocab(vocab_path)
    # with open(corpus_path, 'r', encoding='utf-8') as f:
    with open(corpus_path, 'r', encoding='gbk') as f:
        corpus = f.read()
    
    # 创建模型
    vocab_size = len(tokenizer)
    model = BertAutoRegressiveModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=4,  # 减少层数以加快训练
        num_heads=8
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    print("开始训练BERT自回归语言模型...")
    
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        
        for batch_idx in range(0, train_sample, batch_size):
            # 构建批数据
            x, y = build_bert_dataset(
                min(batch_size, train_sample - batch_idx),
                tokenizer,
                window_size,
                corpus
            )
            
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            
            # 创建注意力掩码（忽略padding）
            attention_mask = (x != 0).long()
            
            # 前向传播
            loss = model(x, attention_mask=attention_mask, labels=y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / (train_sample / batch_size)
        print(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}")
        
        # 生成示例文本
        model.eval()
        prompt = "今天天气"
        generated = generate_text_with_bert(model, tokenizer, prompt, max_length=30)
        print(f"生成文本: {generated}")
    
    # 保存模型
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'config': model.bert.config
        }, save_path)
        print(f"模型已保存到 {save_path}")

if __name__ == "__main__":
    # 需要准备两个文件：
    # 1. corpus.txt: 训练语料
    # 2. vocab.txt: BERT格式的词表文件
    
    # 可以使用预训练的BERT词表，或从头训练
    train_bert_model("corpus.txt", "vocab.txt")
