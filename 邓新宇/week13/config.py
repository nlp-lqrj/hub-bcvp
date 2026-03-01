# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "week09/ner_data/schema.json",
    "train_data_path": "week09/ner_data/train",
    "valid_data_path": "week09/ner_data/test",
    "vocab_path":"week09/chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "use_bert": True,
    "class_num": 9,
    "bert_path": "E:\\BaiduNetdiskDownload\\bert-base-chinese",
    "tuning_tactics": "lora_tuning",
    "seed": 987
}

