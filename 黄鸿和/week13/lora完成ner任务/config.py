# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train",
    "valid_data_path": "data/dev",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 128,  # NER任务通常需要更长的序列长度
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 32,  # NER任务batch size可以适当减小
    "tuning_tactics":"lora_tuning",
    # "tuning_tactics":"p_tuning",
    # "tuning_tactics":"prompt_tuning",
    # "tuning_tactics":"prefix_tuning",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"F:\pretrain_models\bert-base-chinese",
    "seed": 987,
    "class_num": 9  # B-LOCATION, B-ORGANIZATION, B-PERSON, B-TIME, I-LOCATION, I-ORGANIZATION, I-PERSON, I-TIME, O
}