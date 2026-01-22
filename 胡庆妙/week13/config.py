# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_param_path": "output",
    "pretrain_model_path": r"D:/models/bert-base-chinese",

    "schema_path": "data/schema.json",
    "train_data_path": "data/train.txt",
    "valid_data_path": "data/valid.txt",
    "test_data_path": "data/test.txt",

    "sentence_len": 90,

    "epoch_num": 20,  # 训练轮数
    "batch_size": 16,  # 每批的训练样本数
    "learning_rate": 0.0004,
    "optimizer": "adam",
}
