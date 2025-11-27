import torch
import json
import numpy as np
import torch.nn as nn
from transformers import BertModel

'''

通过Pytorch框架实现Bert结构

参数列表：
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


class BertTorchModel(nn.Module):
    #将bert模型配置参数导入
    def __init__(self, config, state_dict=None):
        super(BertTorchModel, self).__init__()
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        self.layer_norm_eps = config['layer_norm_eps']  # 1e-12
        self.max_position_embeddings = config['max_position_embeddings']
        self.num_attention_heads = config['num_attention_heads']    # 12
        self.num_layers = 6   #注意这里的层数要跟预训练config.json文件中的模型层数一致（由原来的num_hidden_layers=12修改为6）
        self.pad_token_id = config['pad_token_id']  # 0
        self.type_vocab_size = config['type_vocab_size']  # 2
        self.vocab_size = config['vocab_size']  # 21128
        # self.load_weights(state_dict)  # 先不加载权重，仅示意模型结构
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)  # 64
        # self.all_head_size = self.num_attention_heads * self.attention_head_size  # 这里考虑 hidden_size无法整除num_attention_heads，来自bert源码

        # 1. bert embedding，使用3层叠加，再经过一个Layer norm层
        self.word_embeddings_layer = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_token_id)
        self.position_embeddings_layer = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        self.token_type_embeddings_layer = nn.Embedding(self.type_vocab_size, self.hidden_size)
        self.embedding_layer_norm_layer = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)


        # 2. bert encoder(transformer)，后使用6层叠加
        # self-attention 层
        # W_q, W_k, W_v 三个线性变换矩阵
        self.query_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.softmax = nn.functional.softmax
        # W_o 线性变换矩阵
        self.attention_output_layer = nn.Linear(self.hidden_size, self.hidden_size)

        # attention layer norm层
        self.attention_output_layer_norm_layer = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # feed forward层
        self.intermediate_dense_layer = nn.Linear(self.hidden_size, self.intermediate_size)
        self.gelu = nn.functional.gelu
        self.output_dense_layer = nn.Linear(self.intermediate_size, self.hidden_size)

        # feed forward layer norm层
        self.output_layer_norm_layer = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)


        # 3. bert pooler层
        self.pooler_dense_layer = nn.Linear(self.hidden_size, self.hidden_size)


    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x):
        # embedding层
        x_embedding = self.embedding_forward(x)   # shape: [batch_size, max_len, hidden_size]
        # transformer层，多层
        sequence_output = self.all_transformer_layer_forward(x_embedding)   # shape: [batch_size, max_len, hidden_size]
        # pooler层
        pooler_output = self.pooler_dense_layer(sequence_output[:, 0, :])  # sequence_output[:, 0] 是沿第 1 维度切片，保留第 0 维和第 2 维
        return sequence_output, pooler_output

    def embedding_forward(self, x):
        batch_size, max_len = x.shape

        # x.shape = [batch_size, max_len]
        we = self.word_embeddings_layer(x)  # shape: [batch_size, max_len, hidden_size]

        # 创建位置索引 [0, 1, 2, ..., max_len-1]
        position_ids = torch.arange(max_len, dtype=torch.long)
        # .unsqueeze(0) 在第0维增加一个维度，从一维变成二维。形状从 [4] 变成 [1, 4]
        # .expand(batch_size, -1)  第一个维度扩展到 batch_size，第二个维度用 -1 表示保持原样
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # position embedding的输入 [0, 1, 2, 3]
        # position_ids.shape: [batch_size, max_len]
        pe = self.position_embeddings_layer(position_ids)  # shape: [batch_size, max_len, hidden_size]

        token_type_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        # token_type_ids.shape: [batch_size, max_len]
        te = self.token_type_embeddings_layer(token_type_ids)  # shape: [batch_size, max_len, hidden_size]
        embedding = we + pe + te
        # 加和后有一个归一化层
        embedding = self.embedding_layer_norm_layer(embedding)  # shape: [batch_size, max_len, hidden_size]
        return embedding

    def all_transformer_layer_forward(self, x):
        # 多层transformer
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x)
        return x

    def single_transformer_layer_forward(self, x_embedding):
        # self attention层
        x_attention = self.self_attention(x_embedding)

        # attention layerNorm
        x_attention = self.attention_output_layer_norm_layer(x_embedding + x_attention)

        # feed forward层
        x_forward = self.intermediate_dense_layer(x_attention)
        x_forward = self.gelu(x_forward)
        x_forward = self.output_dense_layer(x_forward)

        # feed forward layerNorm
        encoder_output = self.output_layer_norm_layer(x_attention + x_forward)
        return encoder_output

    def self_attention(self, x):
        q = self.query_layer(x)  # shape: [batch_size, max_len, hidden_size]
        k = self.key_layer(x)    # shape: [batch_size, max_len, hidden_size]
        v = self.value_layer(x)  # shape: [batch_size, max_len, hidden_size]

        batch_size, max_len, hidden_size = x.shape

        # 多头分别计算self-attention
        # q.shape = batch_size, num_attention_heads, max_len, attention_head_size
        q = q.reshape(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # k.shape = batch_size, num_attention_heads, max_len, attention_head_size
        k = k.reshape(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # v.shape = batch_size, num_attention_heads, max_len, attention_head_size
        v = v.reshape(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # 自相关系数矩阵计算（批量矩阵乘）+归一化
        qk = torch.matmul(q, k.transpose(-2, -1))  # qk.shape = batch_size, num_attention_heads, max_len, max_len
        qk /= torch.sqrt(torch.tensor(self.attention_head_size))
        # qk.shape = batch_size, num_attention_heads, max_len, max_len
        qk = self.softmax(qk, dim=-1)  # dim=-1表示在最后一个维度上进行softmax
        # qkv.shape = batch_size, num_attention_heads, max_len, attention_head_size
        qkv = torch.matmul(qk, v)

        # 拼接多头计算结果，融合多个独立子空间特征
        # qkv.shape = batch_size, max_len, hidden_size
        qkv = qkv.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)

        # attention.shape = batch_size, max_len, hidden_size
        attention = self.attention_output_layer(qkv)
        # key_layer = key_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # value_layer = value_layer.view(
        #     batch_size, -1, self.num_attention_heads, self.attention_head_size
        # ).transpose(1, 2)
        return attention

def load_config_from_json(file_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return config_dict


if __name__ == "__main__":
    json_path = r"D:\pretrain_models\bert-base-chinese\config.json"
    config = load_config_from_json(json_path)

    print("=================Bert参数配置===================")
    for key, value in config.items():
        print(f"{key}: {value}")


    bert = BertModel.from_pretrained(r"D:\pretrain_models\bert-base-chinese", return_dict=False)
    state_dict = bert.state_dict()
    bert.eval()
    x = np.array([[2450, 15486, 102, 2110],
                  [2450, 15486, 102, 2110]])   #假想成4个字的句子
    torch_x = torch.LongTensor(np.array(x))          #pytorch形式输入
    # 不加"[]"维度是一维max_len，(4,)，加了"[]"维度是二维batch_size * max_len，(1, 4)
    # torch_x = torch.LongTensor(np.array(x))

    #自制
    btm = BertTorchModel(config, state_dict)
    diy_sequence_output, diy_pooler_output = btm.forward(torch_x)
    #torch
    torch_sequence_output, torch_pooler_output = bert(torch_x)
    # print(bert.state_dict().keys())  #查看所有的权值矩阵名称

    print(diy_sequence_output, diy_sequence_output.shape)  # torch.Size([2, 4, 768])
    print(torch_sequence_output, torch_sequence_output.shape)  # torch.Size([2, 4, 768])

    print(diy_pooler_output, diy_pooler_output.shape)  #  torch.Size([2, 768])
    print(torch_pooler_output, torch_pooler_output.shape)   #  torch.Size([2, 768])
