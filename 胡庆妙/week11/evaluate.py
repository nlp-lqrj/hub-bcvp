# -*- coding: utf-8 -*-
import random
import sys

import torch
from transformers import BertTokenizer

import loader
from collections import defaultdict

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.logger = logger
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.tokenizer.vocab[" "] = len(self.tokenizer.vocab)  # 在词表中增加空格符
        self.model = model

        self.valid_data = loader.load_valid_data(config["valid_data_path"], config, batch_size=10)
        self.stats_dict = defaultdict(int)  # 用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始验证 第%d轮 的训练效果：" % epoch)
        self.stats_dict = defaultdict(int)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        for index, batch_data in enumerate(self.valid_data):
            batch_ask, batch_ans = batch_data  #
            for ask, ans in zip(batch_ask, batch_ans):
                pred_ans = generate_answer(ask, self.tokenizer, self.model)
                print("问>> ：", ask)
                print("答<< ：", ans)
                print("预测答：", pred_ans)
                print("=" * 30)
                break


def generate_answer(ask, tokenizer: BertTokenizer, model):
    """
    Args:
        ask: 问答系统的问句
        tokenizer: 分词器
        model: 模型
    Returns:
        问答系统的答句
    """
    input_seq = [tokenizer.cls_token_id] + tokenizer.encode(ask, add_special_tokens=False) + [tokenizer.sep_token_id]
    output_max_len = 100  # 输出的最大长度
    output_text = ""
    with torch.no_grad():
        # 生成终止条件1：超过最大输出长度
        while len(output_text) <= output_max_len:
            input_ids = torch.LongTensor([input_seq])
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            logits = model(input_ids)[0][-1]  # 最后一个token的未规范化的预测值

            # 根据采样策略选择预测的字
            vocab_idx = sampling_strategy(logits)
            if vocab_idx == tokenizer.sep_token_id:  # 生成终止条件2：预测字符出现[SEP]
                break
            input_seq.append(vocab_idx)
            output_text += tokenizer.decode([vocab_idx])
    return output_text.replace(tokenizer.pad_token, "")


def sampling_strategy(logits, temperature=0.8):
    prob_distribution = torch.softmax(logits / temperature, dim=0)
    # print("概率分布：", prob_distribution)

    if random.random() > 0.05:
        return int(torch.argmax(prob_distribution))
    else:
        # 从概率最大的K个token中随机选取
        _, idxes = torch.topk(prob_distribution, k=3)
        return random.choice(idxes.tolist())
