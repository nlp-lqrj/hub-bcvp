# -*- coding:utf-8 -*-

# 基础配置
Config = {
    "pretrain_model_path": "D:/AI/models/bert-base-chinese",
    "vocab_path": "D:/AI/models/bert-base-chinese/vocab.txt",
    "model_save_dir": "./peft_model",
    "train_data_path": "./data/train.txt",
    "test_data_path": "./data/test.txt",
    "label_num": 9,
    "num_epochs": 10,
    "batch_size": 20,
    "learning_rate": 5e-4,
    "max_length": 50,
    "use_crf": True,
    "dropout": 0.1,
    "tuning_type": "lora",  # "prompt" "prefix"
    "F1_type": 0     # 计算F1的方式：0为微观，1为宏观
}

# 所有标签
Labels = {
    "O": 0,
    "B-PERSON": 1,
    "I-PERSON": 2,
    "B-LOCATION": 3,
    "I-LOCATION": 4,
    "B-ORGANIZATION": 5,
    "I-ORGANIZATION": 6,
    "B-TIME": 7,
    "I-TIME": 8
}