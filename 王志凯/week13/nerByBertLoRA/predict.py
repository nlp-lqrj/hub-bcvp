# -*- coding:utf-8 -*-

import os
import torch
from config import Config, Labels
from loader import sentence2sequence, load_vocab
from model import NerClassificationModel
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig
cuda = torch.cuda.is_available()


def load_model(config):
    model = NerClassificationModel(config)
    model_dir = config["model_save_dir"]
    tuning_type = config["tuning_type"]
    if tuning_type == "lora":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "attention.output.dense"],
            task_type="TOKEN_CLS"
        )
        peft_model_path = os.path.join(model_dir, "lora.pth")
    elif tuning_type == "prompt":
        peft_config = PromptTuningConfig(
            task_type="TOKEN_CLS",
            num_virtual_tokens=10  # 软提示向量的数量（长度）
        )
        peft_model_path = os.path.join(model_dir, "prompt.pth")
    else:
        peft_config = PrefixTuningConfig(
            task_type="TOKEN_CLS",
            num_virtual_tokens=10  # 每层前缀向量的数量
        )
        peft_model_path = os.path.join(model_dir, "prefix.pth")

    # 将peft配置嵌入模型中
    model = get_peft_model(model, peft_config)
    # 加载peft权重信息
    peft_weight = torch.load(peft_model_path)

    # 将peft权重信息更新到model中
    state_dict = model.state_dict()
    state_dict.update(peft_weight)
    model.load_state_dict(state_dict, strict=False)
    return model

def match_entity(sentences, pred_labels):
    assert len(sentences) == len(pred_labels)
    new_sentences = []
    Schema = {v: k for k, v in Labels.items()}
    for text, labels in zip(sentences, pred_labels):
        labels = labels[: len(text)]
        new_text = ""
        for char, label in zip(text, labels):
            # 给每个字符加上预测标签
            new_text += char + "(" + Schema[int(label)] + ")"
        new_sentences.append(new_text)
    return new_sentences

def predict(sentences):
    model = load_model(Config)
    vocab_dict = load_vocab(Config["vocab_path"])
    sequences = []
    for text in sentences:
        seq = sentence2sequence(text, vocab_dict, Config["max_length"])
        sequences.append(seq)
    sequences = torch.LongTensor(sequences)
    pred_labels = model(sequences)
    print(f"pred_labels:{pred_labels}")
    if not Config["use_crf"]:
        pred_labels = torch.argmax(pred_labels, dim=-1)
    return match_entity(sentences, pred_labels)


if __name__ == '__main__':
    texts = ["马丁路德金在演讲台上说：‘今天我有一个梦想’",
             "2001年9月11日，美国金融大厦发生恐怖袭击",
             "2008年奥巴马成为美国历史上第一位非裔总统",
             "9月3日那天，习近平在北京天安门广场观看了盛大的阅兵仪式"]
    pred_texts = predict(texts)
    for o_text, n_text in enumerate(texts, pred_texts):
        print("="*20)
        print(f"原文：{o_text}")
        print(f"预测：{n_text}")