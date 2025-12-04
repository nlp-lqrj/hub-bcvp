"""
使用torch手动实现BERT
"""

import torch
import torch.nn as nn
import numpy as np
import random
import re
from transformers import BertTokenizer
from collections import defaultdict
from itertools import islice

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


class DiyBertModel(nn.Module):
    def __init__(self, vocab_len, feature_dim):
        super(DiyBertModel, self).__init__()
        # bert的Embedding层
        self.word_emb_layer = nn.Embedding(vocab_len, feature_dim, padding_idx=0)
        self.seg_emb_layer = nn.Embedding(2, feature_dim)
        self.pos_emb_layer = nn.Embedding(512, feature_dim)

        # Transformer Encoding层
        self.q_linear = nn.Linear(feature_dim, feature_dim)  # 计算Q的线性层
        self.k_linear = nn.Linear(feature_dim, feature_dim)  # 计算K的线性层
        self.v_linear = nn.Linear(feature_dim, feature_dim)  # 计算V的线性层
        self.head_concat_linear = nn.Linear(feature_dim, feature_dim)  # 拼接多头之后，会经过一个线性层来融合各个头信息
        self.trans_layerNorm = nn.LayerNorm(feature_dim)  # 残差后的归一化层

        # Feed Forward层
        self.ff_linear_1 = nn.Linear(feature_dim, 4 * feature_dim)  # 前馈网络的第一个线性层，会将特征维度放大4倍，然后过激活层GELU
        self.gelu = nn.GELU(approximate="tanh")
        self.ff_linear_2 = nn.Linear(4 * feature_dim, feature_dim)  # 第二个线性层，将维度转回初始维度（768）
        self.ff_layerNorm = nn.LayerNorm(feature_dim)  # 残差后的归一化层

        # 针对MLM任务的线性层（直接使用 word_emb_layer中的权重）
        self.mlm_linear = nn.Linear(feature_dim, vocab_len)  # 进行MLM分类时，将维度映射到词表大小
        self.mlm_linear.weight = self.word_emb_layer.weight  # 权重共享

        # 针对NSP任务的网络层
        self.pooler_layer = nn.Linear(feature_dim, feature_dim)  # 池化层（取出隐藏状态中的cls，然后过一个线性层和tanh）
        self.tanh = nn.Tanh()
        self.nsp_linear = nn.Linear(feature_dim, 2)  # nsp是二分类任务，所以映射到2维
        # 交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()

    # mlm_label, nsp_label分别为MLM和NSP的真实标签，num_heads为多头数
    def forward(self, x_word_token, x_segment_token, x_position_token, num_layers=12, num_heads=12, mlm_label=None,
                nsp_label=None):
        # embedding层
        word_emb = self.word_emb_layer(x_word_token)
        seg_emb = self.seg_emb_layer(x_segment_token)
        pos_emb = self.pos_emb_layer(x_position_token)
        x = word_emb + seg_emb + pos_emb  # shape(batch_size, len_seq, feature_dim)
        for i in range(num_layers):
            # self_attention层
            q = self.q_linear(x)
            k = self.k_linear(x)
            v = self.v_linear(x)
            # 多头机制
            qkv = self.calc_multi_head_attention(q, k, v, num_heads)
            # 残差和归一化
            z = self.head_concat_linear(qkv)
            x_attention = self.trans_layerNorm(x + z)
            # 前馈神经网络层
            ff_1 = self.ff_linear_1(x_attention)
            gelu = self.gelu(ff_1)
            ff_2 = self.ff_linear_2(gelu)
            # 这里的x即为transformer encoder层最终的输出
            x = self.ff_layerNorm(x_attention + ff_2)  # shape(batch_size, len_seq, feature_dim)
        # 处理 MLM和 NSP任务
        last_hidden_state = x
        if mlm_label is None and nsp_label is None:  # 不传真实标签则直接返回最终隐藏状态
            return last_hidden_state
        # 计算mlm任务的loss
        mask_positions_bool = (mlm_label != -100)  # 将类似[-100,-100,1264,-100]转化为[false,false,true,false]
        mlm_y_true = mlm_label[mask_positions_bool]  # 取出所有真实标签（即被mask掩盖的真实词表索引）
        x_masks = last_hidden_state[mask_positions_bool]  # 取出隐藏状态中mask位置的向量
        mlm_y_pred = self.mlm_linear(x_masks)
        loss_mlm = self.loss(mlm_y_pred, mlm_y_true)
        # 计算nsp任务loss
        cls = last_hidden_state[:, 0, :]  # 从last_hidden_state取出cls：shape(batch_size, feature_dim)
        pooler = self.tanh(self.pooler_layer(cls))  # 池化层
        nsp_y_pred = self.nsp_linear(pooler)
        loss_nsp = self.loss(nsp_y_pred, nsp_label)
        # 两个任务的loss相加进行反向传播
        return loss_mlm + loss_nsp

    # 计算多头注意力值，最后再"拼接"多头
    def calc_multi_head_attention(self, q, k, v, num_heads):
        batch_size, len_seq, feature_dim = q.shape
        multi_dim = int(feature_dim / num_heads)
        # 分别对q k v进行多头切分
        q_temp = q.reshape(batch_size, len_seq, num_heads, multi_dim)
        q_i = q_temp.swapaxes(1, 2)  # shape(batch_size, num_heads, len_seq, multi_dim)
        k_temp = k.reshape(batch_size, len_seq, num_heads, multi_dim)
        k_i = k_temp.swapaxes(1, 2)  # shape(batch_size, num_heads, len_seq, multi_dim)
        v_temp = v.reshape(batch_size, len_seq, num_heads, multi_dim)
        v_i = v_temp.swapaxes(1, 2)  # shape(batch_size, num_heads, len_seq, multi_dim)
        k_i_T = k_i.transpose(2, 3)
        # 计算注意力值
        qk_i = torch.softmax(torch.matmul(q_i, k_i_T) / np.sqrt(multi_dim), dim=-1)
        qkv_i = torch.matmul(qk_i, v_i)  # shape(batch_size, num_heads, len_seq, multi_dim)
        qkv = qkv_i.swapaxes(1, 2).reshape(batch_size, len_seq, -1)  # shape(batch_size, len_seq, feature_dim)
        return qkv


# 随机对文本进行掩码替换（为避免复杂化，直接在两个句子中各替换1个）
def random_mask(word_token):
    spec_ids = tokenizer.convert_tokens_to_ids(["[SEP]", "[CLS]"])
    spec_ids.append(0)
    # 去掉特殊token，包括padding
    new_word_token = [x for x in word_token if x not in spec_ids]
    # 随机取两个字的索引
    nums = random.sample(range(0, len(new_word_token)), 2)
    # 获取对应位置的词表索引
    pos_1 = new_word_token[nums[0]]
    pos_2 = new_word_token[nums[1]]
    pos_ids = [pos_1, pos_2]
    # 保留真实标签，其他全部置为-100
    mask_token = [x if x in pos_ids else -100 for x in word_token]
    # 用mask对应的token替换到对应位置
    mask_id = tokenizer.convert_tokens_to_ids(["[MASK]"])  # 返回列表对象，如果只有一个元素，则取第一个
    word_token = [mask_id[0] if x in pos_ids else x for x in word_token]
    return word_token, mask_token


# 将句子对解析成三个token
def parse_sentence(sentence1, sentence2):
    tokens = tokenizer(sentence1, sentence2, padding="max_length", truncation=True, max_length=32)
    word_token = tokens["input_ids"]
    segment_token = tokens["token_type_ids"]
    position_token = [i for i in range(len(word_token))]
    return word_token, segment_token, position_token


# 将文本转化为序列（为了简化处理文本的代码，使用的标注样本）
def corpus_to_token(sample_size, corpus_path):
    corpus_dict = defaultdict(dict)
    with open(corpus_path, "r", encoding="utf8") as f:
        for line in f:
            # 使用空格切割成三部分，前面两部分为句子对，第三部分为nsp任务的真实标签
            parts = line.split()
            word_token, segment_token, position_token = parse_sentence(parts[0], parts[1])
            is_next = int(parts[2])
            # 随机对句子进行[mask]替换，并返回掩码文本和mask位置的词表索引
            word_token, mask_token = random_mask(word_token)
            # 组装到字典
            corpus_dict[line]["word_token"] = word_token
            corpus_dict[line]["segment_token"] = segment_token
            corpus_dict[line]["position_token"] = position_token
            corpus_dict[line]["mask_token"] = mask_token
            corpus_dict[line]["is_next"] = is_next
            # sample_size不能太大，否则计算超时
            if len(corpus_dict) == sample_size:
                break
    return corpus_dict


# 计算词表大小
def count_vocab(vocab_path):
    count = 0
    with open(vocab_path, "r", encoding="utf8") as f:
        for line in f:
            count += 1
    return count


# 构建batch样本
def build_batch_sample(j, corpus_dict, batch_size):
    x_word_token, x_segment_token, x_position_token, mlm_label, nsp_label = [], [], [], [], []
    start, end = j * batch_size, (j + 1) * batch_size
    if start >= len(corpus_dict):
        return
    if end >= len(corpus_dict):
        end = len(corpus_dict)
    # 使用islice对字典进行切片
    batch_dict = dict(islice(corpus_dict.items(), start, end))
    for key, token_dict in batch_dict.items():
        word_token = token_dict["word_token"]
        segment_token = token_dict["segment_token"]
        position_token = token_dict["position_token"]
        mask_token = token_dict["mask_token"]
        is_next = token_dict["is_next"]
        x_word_token.append(word_token)
        x_segment_token.append(segment_token)
        x_position_token.append(position_token)
        mlm_label.append(mask_token)
        nsp_label.append(is_next)
    x_word_token = torch.tensor(x_word_token)
    x_segment_token = torch.tensor(x_segment_token)
    x_position_token = torch.tensor(x_position_token)
    mlm_label = torch.tensor(mlm_label)
    nsp_label = torch.tensor(nsp_label)
    return x_word_token, x_segment_token, x_position_token, mlm_label, nsp_label


def train(corpus_dict, vocab_len):
    epoch_num = 5  # 训练总轮数
    batch_size = 20  # 20个样本更新一次权重
    feature_dim = 768  # 特征维度
    num_layers = 1  # transformer encoder层数
    num_heads = 12  # 多头数
    lr_rate = 0.001  # 学习率
    # 初始化模型和优化器
    model = DiyBertModel(vocab_len, feature_dim)
    optim = torch.optim.Adam(model.parameters(), lr=lr_rate)
    model.train()
    for i in range(epoch_num):
        watch_loss = []
        for j in range(len(corpus_dict) // batch_size):
            # 构建batch数据集：mlm_label和nsp_label分别表示bert两个子任务的真实标签
            # mlm_label存放每个样本的mask位置索引（词表中的索引），nsp_label存放每个样本的isNext值（1：isNext，0：isNotNext）
            x_word_token, x_segment_token, x_position_token, mlm_label, nsp_label = build_batch_sample(j, corpus_dict,
                                                                                                       batch_size)
            loss = model(x_word_token, x_segment_token, x_position_token, num_layers, num_heads, mlm_label, nsp_label)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print(f"第{i + 1}轮训练的平均loss值为：{np.mean(watch_loss)}")
    # 保存模型
    torch.save(model.state_dict(), "diy_bert_model.pth")


# 预测
def predict(sentence, vocab_len):
    # 加载模型
    feature_dim = 768
    model = DiyBertModel(vocab_len, feature_dim)
    model.load_state_dict(torch.load("diy_bert_model.pth"))
    # 处理句子
    parts = re.split(r"[,，]", sentence, maxsplit=1)
    word_token, segment_token, position_token = parse_sentence(parts[0], parts[1])
    # 因为测试集就一条数据，需要保留batch_size维度，否则和训练时的数据维度不一致
    word_token = torch.tensor([word_token])
    segment_token = torch.tensor([segment_token])
    position_token = torch.tensor([position_token])
    model.eval()
    last_hidden_state = model.forward(word_token, segment_token, position_token)
    print(last_hidden_state.shape)
    print(last_hidden_state[:, 0, :])


if __name__ == "__main__":
    sample_size = 5000  # 样本总数
    # corpus_dict：{sent:{word_token: , segment_token: , position_token: , mask_ids: , is_next: }...}
    corpus_dict = corpus_to_token(sample_size, "corpus.txt")
    # 计算bert词表大小
    vocab_len = count_vocab("bert-base-chinese\\vocab.txt")
    # train(corpus_dict, vocab_len)

    # 使用模型进行预测（输出最终隐藏状态）
    sentence = "深度学习是机器学习的一个小分支，而机器学习是人工智能一个领域"
    predict(sentence, vocab_len)
