import torch
import math
import torch.nn as nn
from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r"D:\W11\Python\第06周 语言模型\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
x = torch.tensor([2450, 15486, 102, 2110])   #假想成4个字的句子
torch_x = x.unsqueeze(0)          #pytorch形式输入
seqence_output, pooler_output = bert(torch_x)
print(seqence_output.shape, pooler_output.shape)
# print(seqence_output, pooler_output)

print(bert.state_dict().keys())  #查看所有的权值矩阵名称

class DiyBert:
    #将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1        #注意这里的层数要跟预训练config.json文件中的模型层数一致
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"]
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"]
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"]
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"]
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"]
        self.transformer_weights = []
        #transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i]
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i]
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i]
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i]
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i]
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i]
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i]
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i]
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i]
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i]
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i]
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i]
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i]
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i]
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i]
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i]
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"]
        self.pooler_dense_bias = state_dict["pooler.dense.bias"]


    #bert embedding，使用3层叠加，在经过一个Layer norm层
    def embedding_forward(self, x):
        # x.shape = [max_len]
        we = self.word_embeddings[x]  # shape: [max_len, hidden_size]
        # position embeding的输入 [0, 1, 2, 3]
        pe = self.position_embeddings[torch.arange(len(x),dtype=torch.long)]  # shpae: [max_len, hidden_size]
        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        te = self.token_type_embeddings[torch.zeros(len(x),dtype=torch.long)]  # shpae: [max_len, hidden_size]
        embedding = we + pe + te
        # 加和后有一个归一化层
        embedding = nn.functional.layer_norm(embedding, [self.hidden_size], self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias,eps=1e-12)
        # embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)  # shpae: [max_len, hidden_size]
        return embedding

    #执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    #执行单层transformer层计算
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        #取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        #self attention层
        q = nn.functional.linear(x,q_w,q_b)
        k = nn.functional.linear(x,k_w,k_b)
        v = nn.functional.linear(x,v_w,v_b)
        empty_w = torch.eye(self.hidden_size,dtype=torch.float)
        empty_b = torch.zeros(self.hidden_size,dtype=torch.float)
        # 调用torch多头attention函数，跳过QKV投影
        attn_output, attn_weights = nn.functional.multi_head_attention_forward(
            q, k, v,
            embed_dim_to_check=self.hidden_size,num_heads=self.num_attention_heads,
            out_proj_weight=attention_output_weight,out_proj_bias=attention_output_bias,
            use_separate_proj_weight=True,
            add_zero_attn=False,dropout_p=0.0,training=False,
            in_proj_weight=None,in_proj_bias=None,bias_k=None,bias_v=None,
            q_proj_weight=empty_w,k_proj_weight=empty_w,v_proj_weight=empty_w
        )
        #bn层，并使用了残差机制
        x = nn.functional.layer_norm(attn_output + x,[self.hidden_size],attention_layer_norm_w,attention_layer_norm_b,1e-12)

        #feed forward层
        feed_forward_x = self.feed_forward(x,
                              intermediate_weight, intermediate_bias,
                              output_weight, output_bias)
        #bn层，并使用了残差机制
        x = nn.functional.layer_norm(x + feed_forward_x,[self.hidden_size], ff_layer_norm_w, ff_layer_norm_b,1e-12)
        return x

    #前馈网络的计算
    def feed_forward(self,
                     x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        x = nn.functional.linear(x, intermediate_weight, intermediate_bias)
        x = nn.functional.gelu(x)
        x = nn.functional.linear(x, output_weight,output_bias)
        return x

    #链接[cls] token的输出层
    def pooler_output_layer(self, x):
        x = nn.functional.linear(x, self.pooler_dense_weight.T, self.pooler_dense_bias)
        x = nn.functional.tanh(x)
        return x

    #最终输出
    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output


#自制
db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)

# print(diy_pooler_output)
# print(torch_pooler_output)
