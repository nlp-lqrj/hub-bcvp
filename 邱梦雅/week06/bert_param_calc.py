import json

'''
attention_probs_dropout_prob = 0.1
directionality ="bidi"
hidden_act = "gelu"
hidden_dropout_prob = 0.1
hidden_size = 768
initializer_range = 0.02
intermediate_size = 3072
layer_norm_eps = 1e-12
max_position_embeddings = 512
model_type = "bert"
num_attention_heads = 12
num_hidden_layers = 6
pad_token_id = 0,
pooler_fc_size = 768
pooler_num_attention_heads = 12
pooler_num_fc_layers = 3
pooler_size_per_head = 128
pooler_type = "first_token_transform"
type_vocab_size = 2
vocab_size = 21128
'''

def load_config_from_json(file_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return config_dict


if __name__ == "__main__":
    json_path = r"D:\pretrain_models\bert-base-chinese\config.json"
    config_dict = load_config_from_json(json_path)

    print("=================参数配置===================")
    for key, value in config_dict.items():
        print(f"{key}: {value}")

    print("=================参数统计===================")

    embedding_param_size = 0
    transformer_param_size = 0
    pooler_param_size = 0
    bert_param_size = 0
    '''
    Embedding部分参数计算
    (vocab_size + max_position_embeddings + type_vocab_size + 2) * hidden_size 
    '''
    # embeddings.word_embeddings.weight
    # vocab_size, embedding_dim（hidden_size）
    embedding_param_size += config_dict["vocab_size"] * config_dict["hidden_size"]

    # embeddings.position_embeddings.weight
    # max_position_embeddings, embedding_dim（hidden_size）
    embedding_param_size += config_dict["max_position_embeddings"] * config_dict["hidden_size"]

    # embeddings.token_type_embeddings.weight
    # type_vocab_size, embedding_dim（hidden_size）
    embedding_param_size += config_dict["type_vocab_size"] * config_dict["hidden_size"]

    # embeddings.LayerNorm.weight
    # embedding_dim（hidden_size）,
    embedding_param_size += config_dict["hidden_size"]

    # embeddings.LayerNorm.bias
    # embedding_dim（hidden_size）,
    embedding_param_size += config_dict["hidden_size"]

    '''
    Transformer部分参数计算
    ((hidden_size * hidden_size + hidden_size) * 4 + hidden_size * 2 + intermediate_size * hidden_size + intermediate_size + \
     intermediate_size * hidden_size + hidden_size + hidden_size * 2) * num_hidden_layers
     => (4 * hidden_size * hidden_size + (2 * intermediate_size + 9) * hidden_size + intermediate_size) * num_hidden_layers
    '''
    # num_layers = config_dict["num_hidden_layers"]
    num_layers = 12  # 这里使用默认配置12层
    for i in range(num_layers):
        # encoder.layer.0.attention.self.query.weight
        # hidden_size, hidden_size
        transformer_param_size += config_dict["hidden_size"] * config_dict["hidden_size"]

        # encoder.layer.0.attention.self.query.bias
        # hidden_size,
        transformer_param_size += config_dict["hidden_size"]

        # encoder.layer.0.attention.self.key.weight
        # hidden_size, hidden_size
        transformer_param_size += config_dict["hidden_size"] * config_dict["hidden_size"]

        # encoder.layer.0.attention.self.key.bias
        # hidden_size,
        transformer_param_size += config_dict["hidden_size"]

        # encoder.layer.0.attention.self.value.weight
        # hidden_size, hidden_size
        transformer_param_size += config_dict["hidden_size"] * config_dict["hidden_size"]

        # encoder.layer.0.attention.self.value.bias
        # hidden_size,
        transformer_param_size += config_dict["hidden_size"]

        # encoder.layer.0.attention.output.dense.weight
        # hidden_size, hidden_size
        transformer_param_size += config_dict["hidden_size"] * config_dict["hidden_size"]

        # encoder.layer.0.attention.output.dense.bias
        # hidden_size,
        transformer_param_size += config_dict["hidden_size"]

        # encoder.layer.0.attention.output.LayerNorm.weight
        # hidden_size,
        transformer_param_size += config_dict["hidden_size"]

        # encoder.layer.0.attention.output.LayerNorm.bias
        # hidden_size,
        transformer_param_size += config_dict["hidden_size"]

        # encoder.layer.0.intermediate.dense.weight
        # intermediate_size, hidden_size
        transformer_param_size += config_dict["intermediate_size"] * config_dict["hidden_size"]

        # encoder.layer.0.intermediate.dense.bias
        # intermediate_size,
        transformer_param_size += config_dict["intermediate_size"]

        # encoder.layer.0.output.dense.weight
        # hidden_size, intermediate_size
        transformer_param_size += config_dict["hidden_size"] * config_dict["intermediate_size"]

        # encoder.layer.0.output.dense.bias
        # hidden_size,
        transformer_param_size += config_dict["hidden_size"]

        # encoder.layer.0.output.LayerNorm.weight
        # hidden_size,
        transformer_param_size += config_dict["hidden_size"]

        # encoder.layer.0.output.LayerNorm.bias
        # hidden_size,
        transformer_param_size += config_dict["hidden_size"]

    '''
    pooler层参数计算
    '''
    # pooler.dense.weight
    # hidden_size, hidden_size
    pooler_param_size += config_dict["hidden_size"] * config_dict["hidden_size"]

    # pooler.dense.bias
    # hidden_size
    pooler_param_size += config_dict["hidden_size"]


    bert_param_size = embedding_param_size + transformer_param_size + pooler_param_size

    print(f"Embedding参数大小: {embedding_param_size} 个")  # 16622592 个
    print(f"Transformer参数大小: {transformer_param_size} 个")  # 85054464 个
    print(f"pooler参数大小: {pooler_param_size} 个")   # 590592 个
    print(f"BERT参数大小: {bert_param_size} 个，" f"约等于 {bert_param_size / 1e9:.2f}B 参数")  # BERT参数大小: 102267648 个，约等于 0.10B 参数

    print("=================公式计算===================")
    """
    (4 * num_hidden_layers + 1) * hidden_size * hidden_size
    + (vocab_size + max_position_embeddings + type_vocab_size + 2 * intermediate_size * num_hidden_layers + 9 * num_hidden_layers + 3) * hidden_size
    + intermediate_size * num_hidden_layers
    """
    h = config_dict["hidden_size"]
    vocab_size = config_dict["vocab_size"]
    max_position_embeddings = config_dict["max_position_embeddings"]
    type_vocab_size = config_dict["type_vocab_size"]
    # num_layers = 12
    intermediate_size = config_dict["intermediate_size"]

    bert_param_size_2 = 0
    bert_param_size_2 += (4 * num_layers + 1) * h ** 2 + \
        (vocab_size + max_position_embeddings + type_vocab_size + 2 * intermediate_size * num_layers + 9 * num_layers + 3) * h + \
        intermediate_size * num_layers

    print(f"BERT参数大小: {bert_param_size_2} 个，"
          f"约等于 {bert_param_size_2 / 1e9:.2f}B 参数")