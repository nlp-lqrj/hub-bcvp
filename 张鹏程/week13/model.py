import torch.nn as nn
from config import Config
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from torch.optim import Adam, SGD

def get_model():
    """获取模型，延迟加载"""
    return AutoModelForTokenClassification.from_pretrained(
        Config["pretrain_model_path"],
        num_labels=Config["class_num"]
    )

# 为了兼容原有代码，保留TorchModel变量
TorchModel = None


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)