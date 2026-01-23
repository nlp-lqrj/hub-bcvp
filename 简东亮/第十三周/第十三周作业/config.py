"""
配置参数信息
"""

配置 = {
    "model_path" : "output" ,
    "train_data_path" : "data/train" ,
    "valid_data_path" : "data/dev" ,
    "vocab_path" : "chars.txt" ,
    "model_type" : "bert" ,
    "max_length" : 128 ,    # NER任务通常需要更长的序列长度
    "hidden_​​​​size" : 128 ,
    "kernel_size" : 3 ,
    "num_layers" : 2 ,
    “epoch”：10，
    "batch_size" : 32 ,    # NER任务batch size可以适当减小
    "tuning_tactics" : "lora_tuning" ,
    "pooling_style" : "max" ,
    "优化器" : "亚当" ,
    "learning_rate" : 1e-3 ,
    "pretrain_model_path" : r"F:\pretrain_models\bert-base-chinese" ,
    “种子”：987，
    "class_num" : 9   # B-LOCATION, B-ORGANIZATION, B-PERSON, B-TIME, I-LOCATION, I-ORGANIZATION, I-PERSON, I-TIME, O
}
