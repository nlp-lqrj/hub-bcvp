import torch
import torch.nn as nn
from transformers import BertModel

'''
基于pytorch的自实现的Bert模型, 并且扩展了计算模型参数总量的方法
'''


class BertBaseTorch:
    # 将预训练好的整个权重字典输入进来
    def __init__(self, state_dict, sen_len):
        self.hidden_size = 768  # 词向量的维度，不能改
        self.sen_len = sen_len
        self.num_attention_heads = 12  # Mulit-Head机制要拆分的head数，要跟预训练config.json文件中的num_attention_heads一致
        self.num_hidden_layers = 12  # 跟预训练config.json文件中的num_hidden_layers一致

        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        # embedding层的 3个embedding的参数
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"]  # 词向量编码, [词表大小, 768]
        self.segment_embeddings = state_dict["embeddings.token_type_embeddings.weight"]  # segment编码, [2, 768]
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"]  # 位置编码, [512, 768]

        # embedding层的 归一化参数  [768, 768]
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"]  # shape: [768]
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"]  # shape: [768]

        # transformer层（多层）
        self.transformer_params = []
        for i in range(self.num_hidden_layers):
            # self-attention的  Q,K,V -> softmax(Q * K.T) * V 的参数
            attention_q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i]  # shape: [768, 768]
            attention_q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i]  # shape: [768]
            attention_k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i]
            attention_k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i]
            attention_v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i]
            attention_v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i]

            # self-attention的 Linear(Attention(𝑄,𝐾,𝑉)) 的参数
            attention_output_weight = state_dict[
                "encoder.layer.%d.attention.output.dense.weight" % i]  # shape: [768, 768]
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i]  # shape: [768]

            # self-attention的 归一化层参数
            attention_layer_norm_w = state_dict[
                "encoder.layer.%d.attention.output.LayerNorm.weight" % i]  # shape: [768]
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i]  # shape: [768]

            # feed forward的 linar(gelu(linar(x)))的参数
            ff_intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i]  # shape: [3072,768]
            ff_intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i]  # shape: [3072]
            ff_output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i]  # shape: [768,3072]
            ff_output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i]  # shape: [768]

            # feed forward的 归一化层的的参数
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i]  # shape: [768]
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i]  # shape: [768]
            self.transformer_params.append(
                [attention_q_w, attention_q_b, attention_k_w, attention_k_b, attention_v_w, attention_v_b,
                 attention_output_weight, attention_output_bias,
                 attention_layer_norm_w, attention_layer_norm_b, ff_intermediate_weight, ff_intermediate_bias,
                 ff_output_weight, ff_output_bias, ff_layer_norm_w, ff_layer_norm_b])

        # pooler层 tan(x*W.T + b) 的参数
        self.pooler_dense_weight = state_dict["pooler.dense.weight"]  # shape: [768, 768]
        self.pooler_dense_bias = state_dict["pooler.dense.bias"]  # shape: [768]

    # 计算模型的参数总量
    def cal_params_total(self):
        # embedding层的参数
        ecnt = 0
        ecnt += self.cal_param_of_tensor(self.word_embeddings)  # 词向量编码, [词表大小, 768]
        ecnt += self.cal_param_of_tensor(self.segment_embeddings)  # segment编码, [2, 768]
        ecnt += self.cal_param_of_tensor(self.position_embeddings)  # # 位置编码, [512, 768]
        ecnt += self.cal_param_of_tensor(self.embeddings_layer_norm_weight)  # [768, 768]
        ecnt += self.cal_param_of_tensor(self.embeddings_layer_norm_bias)  # [768]
        print("\tembedding层的参数量：", format(ecnt, ","))

        # transformer层的参数
        acnt = 0
        for item in self.transformer_params[0]:
            cnt = 1
            i = 0
            while (i < len(item.shape)):
                cnt *= item.shape[i]
                i += 1
            acnt += cnt
        acnt *= self.num_hidden_layers
        print("\ttransformer层的参数量：", format(acnt, ","))

        # pooler层 tan(x*W.T + b) 的参数
        pcnt = 0
        pcnt += self.cal_param_of_tensor(self.pooler_dense_weight)  # [768, 768]
        pcnt += self.cal_param_of_tensor(self.pooler_dense_bias)  # [768]
        print("\tpooler层的参数量：", format(pcnt, ","))

        return ecnt + acnt + pcnt

    # 计算指定张量包含参数
    def cal_param_of_tensor(self, tensor):
        cnt = 1
        i = 0
        while (i < len(tensor.shape)):
            cnt *= tensor.shape[i]
            i += 1
        return cnt

    # 最终输出
    def forward(self, x):
        print("\n1. embedding层输入:\n", x)
        # [batch_size, sen_len] -> [batch_size, sen_len, 768]
        embeded_x = self.embedding_layer(x)
        # print("  embedding层输出：\n", embeded_x)

        print("\n2. transformer层输入:\n", embeded_x)
        sequence_output = self.transformer_layers(embeded_x)
        # print("  transformer层输出：\n", sequence_output)

        # sequence_output[0] 表示?
        print("\n3. pooler_output层输入:\n", sequence_output)
        # shape: [batch_size, sen_len, 768) ->  [batch_size, 768]
        pooler_output = self.pooler_output_layer(sequence_output)
        return sequence_output, pooler_output

    # bert embedding，使用3层叠加，在经过一个Layer norm层
    def embedding_layer(self, x):
        batch_size = x.shape[0]

        print("\t1.1. 对x做 word embedding, segment embedding, position embedding ...")
        # [batch_size, sen_len] -> [batch_size, sen_len, 768]
        we = self.get_embedding(self.word_embeddings, x)

        # [batch_size, sen_len] -> [batch_size, sen_len, 768]
        te = self.get_embedding(self.segment_embeddings,
                                torch.LongTensor(batch_size * [self.sen_len * [0]]))

        # [batch_size, sen_len] -> [batch_size, sen_len, 768]
        pe = self.get_embedding(self.position_embeddings,
                                torch.LongTensor(batch_size * [list(range(self.sen_len))]))

        print("\t1.2. 将这三种embedding相加，再做 layer norm ...")
        embedding = we + pe + te

        # [batch_size, sen_len, 768]
        return self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)

    def get_embedding(self, embedding_matrix, x):
        batch_size = x.shape[0]
        vectors = []
        for i in range(batch_size):
            vectors.append(embedding_matrix[x[i]])
        result = torch.stack(vectors, dim=0).squeeze(1)  # 或者直接 dim=0
        return result

    def layer_norm(self, x, layer_norm_w, layer_norm_b):
        batch_size = x.shape[0]
        vectors = []
        for i in range(batch_size):
            mean = torch.mean(x[i], dim=1, keepdim=True)
            std = torch.std(x[i], dim=1, keepdim=True)
            vectors.append((x[i] - mean) / std)
        norm_x = torch.stack(vectors, dim=0).squeeze(1)
        return norm_x * layer_norm_w + layer_norm_b

    # 执行全部的transformer层计算
    def transformer_layers(self, x):
        """
        Args:
            x: shape: [batch_size, sen_len, 768]
        """
        for i in range(self.num_hidden_layers):
            print(f"\ttransformer 第{i + 1}层 的计算...")
            x = self.transformer_layer(x, i)
        return x

    # 执行单层transformer层计算
    def transformer_layer(self, embedding_x, layer_index):
        """
        Args:
            embedding_x: shape: [batch_size, sen_len, 768]
        Returns:
            返回参的 shape: [batch_size, sen_len, 768]
        """
        params = self.transformer_params[layer_index]

        # 取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        attention_q_w, attention_q_b, \
            attention_k_w, attention_k_b, \
            attention_v_w, attention_v_b, \
            attention_output_weight, attention_output_bias, \
            attention_layer_norm_w, attention_layer_norm_b, \
            ff_intermediate_weight, ff_intermediate_bias, \
            ff_output_weight, ff_output_bias, \
            ff_layer_norm_w, ff_layer_norm_b = params

        # self attention 的计算
        print("\t\t2.1. self attention 的计算 ... ")
        attention_x = self.self_attention(embedding_x,
                                          attention_q_w, attention_q_b,
                                          attention_k_w, attention_k_b,
                                          attention_v_w, attention_v_b,
                                          attention_output_weight,
                                          attention_output_bias)

        print("\t\t2.2. 使用残差机制(即：embedding_x + attention_x), 再做layer norm ... ")
        # shape: [batch_size, sen_len, 768] * [768] + [768] -> [batch_size, sen_len, 768]
        attention_normed_x = self.layer_norm(embedding_x + attention_x, attention_layer_norm_w, attention_layer_norm_b)

        # feed forward层
        print("\t\t2.3. feed forward 的计算 ... ")
        feed_forward_x = self.feed_forward(attention_normed_x,
                                           ff_intermediate_weight, ff_intermediate_bias,
                                           ff_output_weight, ff_output_bias)

        print("\t\t2.4. 使用残差机制(即：attention_normed_x + feed_forward_x), 再做layer norm ... ")
        return self.layer_norm(attention_normed_x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)

    def self_attention(self,
                       embeded_x,
                       attention_q_w,
                       attention_q_b,
                       attention_k_w,
                       attention_k_b,
                       attention_v_w,
                       attention_v_b,
                       attention_output_weight,
                       attention_output_bias):
        """
        Args:
            embeded_x: shap: [batch_size, sen_len, 768]
        Returns:
            返回参的shape: [batch_size, sen_len, 768]
        """
        batch_size = embeded_x.shape[0]
        attention_head_size = int(self.hidden_size / self.num_attention_heads)  # Muliti-Head机制每个Head的列数

        attention_vectors = []
        for i in range(batch_size):
            print(f"\t\t\t第 {i} 个批量...")
            print("\t\t\t\t计算 q, k, v， q=linear(x), k=linear(x), v=linear(x) ...")
            # shape: [sen_len, 768] * [768, 768] + [768] -> [sen_len, 768]
            q = torch.matmul(embeded_x[i], attention_q_w.T) + attention_q_b
            k = torch.matmul(embeded_x[i], attention_k_w.T) + attention_k_b
            v = torch.matmul(embeded_x[i], attention_v_w.T) + attention_v_b

            # 拆分 Muliti-Head, shape: [sen_len, 768] -> [sen_len, 12, 64] -> [12, sen_len, 64]
            print("\t\t\t\t将 q, k, v 拆分为多头，相当于: [sen_len, 768] -> [12, sen_len, 768/12] ... ")
            q = q.reshape(self.sen_len, self.num_attention_heads, attention_head_size)
            q = q.transpose(0, 1)
            k = k.reshape(self.sen_len, self.num_attention_heads, attention_head_size)
            k = k.transpose(0, 1)
            v = v.reshape(self.sen_len, self.num_attention_heads, attention_head_size)
            v = v.transpose(0, 1)

            # 计算 softmax(q * k.T/sqrt(head_size)) * v
            print("\t\t\t\t计算 qkv = softmax(q * k.T/sqrt(head_size)) * v ...")
            # q * k.T,  shape: [12, sen_len, 64] * [12, 64, sen_len] -> [12, sen_len, sen_len]
            qk = torch.matmul(q, k.transpose(1, 2))
            qk = torch.softmax(qk / torch.sqrt(torch.LongTensor([attention_head_size])), dim=-1)
            # shape: [12, sen_len, sen_len] * [12, sen_len, 64] -> [12, sen_len, 64]
            qkv = torch.matmul(qk, v)

            # shape: [12, sen_len, 64] -> [sen_len, 12, 64] -> [sen_len, 768]
            qkv = qkv.transpose(0, 1).reshape(-1, self.hidden_size)

            print("\t\t\t\t计算 attention的输出：linear( attention(k,q,v) ) ...")
            # shape: [sen_len, 768] * [768, 768] -> [sen_len, 768]
            attention = torch.matmul(qkv, attention_output_weight.T) + attention_output_bias
            attention_vectors.append(attention)

        return torch.stack(attention_vectors, dim=0)  # shape: [batch_size, sen_len, 768]

    # 前馈网络的计算
    def feed_forward(self,
                     attention_normed_x,
                     intermediate_weight,
                     intermediate_bias,
                     output_weight,
                     output_bias,
                     ):
        """
        Args:
            attention_normed_x: shap: [batch_size, sen_len, 768]
        Returns:
            返回参的shape: [batch_size, sen_len, 768]
        """
        batch_size = attention_normed_x.shape[0]
        sequence_vectors = []
        for i in range(batch_size):
            print(f"\t\t\t第 {i} 个批量...")
            print("\t\t\t\t计算 linear(gelu(linear(x))) ...")
            # shape: [sen_len, 768] * [768, 3072] + [3072] -> [sen_len, 3072]
            tmp = torch.matmul(attention_normed_x[i], intermediate_weight.T) + intermediate_bias

            tmp = nn.GELU()(tmp)

            # shape: [sen_len, 3072] * [3072, 768] + [768] -> [sen_len, 768]
            tmp = torch.matmul(tmp, output_weight.T) + output_bias
            sequence_vectors.append(tmp)
        return torch.stack(sequence_vectors, dim=0)  # shape: [batch_size, sen_len, 768]

    def pooler_output_layer(self, x):
        """
        Args:
            x: shape: [batch_size, sen_len, 768)
        Returns:
            返回参的shape: [batch_size, 768]
        """
        batch_size = x.shape[0]

        pooler_vectors = x[:, 0, :]  # 取每次批次中每句话的每一个token对应的向量
        print("结果形状:", pooler_vectors.shape)  # torch.Size([batch_size, 768])

        # shape: [batch_size, 768] * [768, 768] + [768] -> [batch_size, 768]
        poller_output = torch.matmul(pooler_vectors, self.pooler_dense_weight.T) + self.pooler_dense_bias
        poller_output = torch.tanh(poller_output)
        return poller_output


x = torch.LongTensor([[2450, 15486, 102, 2110], [2450, 15486, 102, 2110]])  # 假想成4个字的句子

bert = BertModel.from_pretrained(r"D:\Miniconda3\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()

sequence_output, pooler_output = bert(x)
print("\nsequence_output:\n", sequence_output, sequence_output.shape)  # shape: [batch_size, sen_len, 768]
print("pooler_output:\n", pooler_output, pooler_output.shape)  # shape: [batch_size, 768]
# print(bert.state_dict().keys())  # 查看所有的参数的名称

mybert = BertBaseTorch(state_dict, 4)
my_sequence_output, my_pooler_output = mybert.forward(x)
print("my_sequence_output: \n", my_sequence_output, my_sequence_output.shape)  # shape: [batch_size, sen_len, 768]
print("my_pooler_output: \n", my_pooler_output, my_pooler_output.shape)  # shape: [batch_size, 768]

print("计算模型参数总量...")
total_params_cnt = mybert.cal_params_total()
print("模型参数总量: ", format(total_params_cnt, ","))
