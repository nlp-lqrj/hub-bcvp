# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "test_data_path": "../data/test.json",
    "vocab_path": "../chars.txt",
    "sentence_len": 20,
    "embed_size": 128,
    "num_epochs": 10,
    "batch_size": 20,
    "epoch_sample_size": 200,  # 每轮训练中采样数量

    "optimizer": "adam",
    "learning_rate": 1e-3,
}
