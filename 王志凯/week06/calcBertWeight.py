"""
计算BERT中所有权重参数个数和参数占用内存大小
"""

import numpy as np
import re
from transformers import BertModel
from collections import defaultdict

# BERT权重信息
class BertWeightInfo():
    def __init__(self, config):
        self.hidden_size = config["hidden_size"]                # 特征维度
        self.position_len = config["max_position_embeddings"]   # position embedding的最大长度
        self.segment_len = config["type_vocab_size"]            # segment embedding 类型个数
        self.num_layers = config["num_hidden_layers"]           # transformer encoder层数
        self.vocab_size = config["vocab_size"]                  # 词表大小
        self.torch_dtype = config["torch_dtype"]                # 存储的参数类型
        self.version = config["transformers_version"]           # 4.50.0及以后的版本把线性层的偏置项都移除了（除了norm层）
        self.weight_information = defaultdict(dict)             # 记录bert各个网络层的参数信息（包括总信息）
        self.calc_total_weight_count()
        self.calc_total_weight_size()

    # 计算参数个数
    def calc_total_weight_count(self):
        # embedding层
        word_token_weight = self.vocab_size * self.hidden_size
        segment_token_weight = self.segment_len * self.hidden_size
        position_token_weight = self.position_len * self.hidden_size
        self.embedding_weight = word_token_weight + segment_token_weight + position_token_weight
        # self-attention
        qkv_weight = 3 * self.hidden_size * self.hidden_size               # 由x转化为 Q K V 的三个线性层
        multihead_weight = self.hidden_size * self.hidden_size             # 多头融合时的线性层
        attention_layerNorm_weight = self.hidden_size * self.hidden_size   # 残差之后的归一化层
        attention_layerNorm_bias = self.hidden_size
        self.attention_weight = self.num_layers * (qkv_weight + multihead_weight + attention_layerNorm_weight + attention_layerNorm_bias)
        # feed-forward
        ff1_linear_weight = self.hidden_size * self.hidden_size * 4        # 第一个线性层（将维度映射到 4H）
        ff2_linear_weight = 4 * self.hidden_size * self.hidden_size        # 第二个线性层（将维度映射回 H）
        ff_layerNorm_weight = self.hidden_size * self.hidden_size          # 残差之后的归一化层
        ff_layerNorm_bias = self.hidden_size
        self.feed_forward_weight = self.num_layers * (ff1_linear_weight + ff2_linear_weight + ff_layerNorm_weight + ff_layerNorm_bias)
        # 判断版本：如果是4.50.0及以前的版本，则所有线性层都有偏置项
        if self.version < "4.50.0":
            qkv_bias = 3 * self.hidden_size
            multihead_bias = self.hidden_size
            ff1_linear_bias = self.hidden_size * 4
            ff2_linear_bias = self.hidden_size
            self.attention_weight += self.num_layers * (qkv_bias + multihead_bias)
            self.feed_forward_weight += self.num_layers * (ff1_linear_bias + ff2_linear_bias)
        # 在对MLM和NSP任务做训练时，需要将维度映射到分类维度，所以也需要全连接层
        # 但是MLM任务需要将维度映射到词表大小，而 word_embedding层的权重正好符合要求，所以进行了权重共享，不需要额外计算参数量
        # 这里只需要计算NSP任务中的池化层和分类层权重，但是分类层的权重属于具体任务参数，bert并没有对其进行开源，所以不算做bert权重范围内
        nsp_pooler_weight = self.hidden_size * self.hidden_size
        nsp_pooler_bias = self.hidden_size
        self.pooler_weight = nsp_pooler_weight + nsp_pooler_bias
        # bert总参数量
        self.total_weight = self.embedding_weight + self.attention_weight + self.feed_forward_weight + self.pooler_weight
        self.weight_information["total"]["weight_count"] = self.total_weight
        self.weight_information["embedding"]["weight_count"] = self.embedding_weight
        self.weight_information["attention"]["weight_count"] = self.attention_weight
        self.weight_information["feedforward"]["weight_count"] = self.feed_forward_weight
        self.weight_information["pooler"]["weight_count"] = self.pooler_weight
        # 参数占比
        self.weight_information["total"]["percent"] = "100%"
        self.weight_information["embedding"]["percent"] = f"{self.embedding_weight/self.total_weight:.2%}"
        self.weight_information["attention"]["percent"] = f"{self.attention_weight/self.total_weight:.2%}"
        self.weight_information["feedforward"]["percent"] = f"{self.feed_forward_weight/self.total_weight:.2%}"
        self.weight_information["pooler"]["percent"] = f"{self.pooler_weight/self.total_weight:.2%}"


    # 计算参数大小（所占字节数）
    def calc_total_weight_size(self):
        float_type = int(re.findall(r'\d+', self.torch_dtype)[0])
        # 一个浮点数占用的字节数
        self.floatByte = int(float_type / 8)
        self.embedding_weight_size = f"{self.embedding_weight * self.floatByte/1024/1024:.2f}M"
        self.attention_weight_size = f"{self.attention_weight * self.floatByte/1024/1024:.2f}M"
        self.feed_forward_weight_size = f"{self.feed_forward_weight * self.floatByte/1024/1024:.2f}M"
        self.pooler_weight_size = f"{self.pooler_weight * self.floatByte/1024/1024:.2f}M"
        self.total_weight_size = f"{self.total_weight * self.floatByte/1024/1024:.2f}M"
        # 参数大小（转化为 M）
        self.weight_information["total"]["weight_size"] = self.total_weight_size
        self.weight_information["embedding"]["weight_size"] = self.embedding_weight_size
        self.weight_information["attention"]["weight_size"] = self.attention_weight_size
        self.weight_information["feedforward"]["weight_size"] = self.feed_forward_weight_size
        self.weight_information["pooler"]["weight_size"] = self.pooler_weight_size


if __name__ == "__main__":
    # 获取bert的配置信息
    bert = BertModel.from_pretrained("bert-base-chinese")
    config = bert.config.to_dict()

    # BertWeightInfo中包含各种权重参数
    weightInfo = BertWeightInfo(config)

    # info包含bert里面所有网络层的参数信息和总参数信息，包括参数个数、参数占比、参数大小
    info = weightInfo.weight_information
    # print(info)

    # 也可以单独查看单一参数信息，比如总数大小
    total_size = weightInfo.total_weight_size
    print(total_size)

    # 查看self-attention和feed forward参数对比
    print(info["attention"])
    print(info["feedforward"])