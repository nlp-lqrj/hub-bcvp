# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "num_layers": 2,
    "epoch": 30,
    "batch_size": 64,
    "tuning_tactics": "lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 2e-4, #2e-5
    "use_crf": True,
    "class_num": 9,
    "pretrain_model_path": r"E:\newlife\badou\第六周 语言模型\bert-base-chinese"
}

