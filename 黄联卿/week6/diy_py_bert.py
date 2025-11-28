import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel


'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r"E:\newlife\badou\第六周 语言模型\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 102, 2110])   #假想成4个字的句子
torch_x = torch.LongTensor([x])          #pytorch形式输入
ori_bert_sequence_output, ori_bert_pooler_output = bert(torch_x)
print("原Bert输出形状：", ori_bert_sequence_output.shape, ori_bert_pooler_output.shape)

print(bert.state_dict().keys())  #查看所有的权值矩阵名称


# PyTorch Bert（复现原numpy功能）
class PyTorchBert(nn.Module):
    def __init__(self, state_dict, num_layers=1):
        super().__init__()
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.num_layers = num_layers
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        # 加载预训练权重（复用state_dict键名）
        self._load_weights(state_dict)

    def _load_weights(self, state_dict):
        # 1. Embedding层
        self.word_embeddings = nn.Embedding.from_pretrained(
            state_dict["embeddings.word_embeddings.weight"], freeze=True
        )
        self.position_embeddings = nn.Embedding.from_pretrained(
            state_dict["embeddings.position_embeddings.weight"], freeze=True
        )
        self.token_type_embeddings = nn.Embedding.from_pretrained(
            state_dict["embeddings.token_type_embeddings.weight"], freeze=True
        )
        self.embedding_ln = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.embedding_ln.weight = nn.Parameter(state_dict["embeddings.LayerNorm.weight"])
        self.embedding_ln.bias = nn.Parameter(state_dict["embeddings.LayerNorm.bias"])

        # 2. Transformer层（多层共享结构，加载对应层权重）
        self.transformer_layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = TransformerLayer(self.hidden_size, self.num_attention_heads)
            # 加载该层权重
            layer.load_weights(state_dict, layer_idx=i)
            self.transformer_layers.append(layer)

        # 3. Pooler层
        self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
        self.pooler.weight = nn.Parameter(state_dict["pooler.dense.weight"])
        self.pooler.bias = nn.Parameter(state_dict["pooler.dense.bias"])

    def forward(self, x):
        """
        输入：x -> numpy数组[seq_len]（如[2450, 15486, 102, 2110]）
        输出：sequence_output[seq_len, 768], pooler_output[768]
        """
        # 转换为 PyTorch tensor
        x = torch.LongTensor(x).unsqueeze(0)  # [1, seq_len]

        # 1. Embedding 层（word + position + token_type）
        seq_len = x.shape[1]
        word_emb = self.word_embeddings(x)  # [1, seq_len, 768]
        pos_emb = self.position_embeddings(torch.arange(seq_len, device=x.device).unsqueeze(0))  # [1, seq_len, 768]
        token_type_emb = self.token_type_embeddings(torch.zeros_like(x))  # [1, seq_len, 768]

        emb = word_emb + pos_emb + token_type_emb
        emb = self.embedding_ln(emb)  # [1, seq_len, 768]

        # 2. Transformer编码
        hidden_states = emb
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states)  # [1, seq_len, 768]

        # 3. 输出处理（去掉batch维度）
        sequence_output = hidden_states.squeeze(0).detach().numpy()  # [seq_len, 768]
        pooler_output = torch.tanh(self.pooler(hidden_states[:, 0, :])).squeeze(0).detach().numpy()  # [768]

        return sequence_output, pooler_output


class TransformerLayer(nn.Module):
    """简化版 Transformer 层（复用预训练权重）"""

    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # 注意力层
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_out_proj = nn.Linear(hidden_size, hidden_size)

        # LayerNorm
        self.attn_ln = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn_ln = nn.LayerNorm(hidden_size, eps=1e-12)

        # 前馈网络
        self.ffn_intermediate = nn.Linear(hidden_size, 3072)  # bert-base 中间层维度 3072
        self.ffn_out = nn.Linear(3072, hidden_size)

    def load_weights(self, state_dict, layer_idx):
        """加载对应层的预训练权重"""
        prefix = f"encoder.layer.{layer_idx}."
        # 注意力层权重
        self.q_proj.weight = nn.Parameter(state_dict[f"{prefix}attention.self.query.weight"])
        self.q_proj.bias = nn.Parameter(state_dict[f"{prefix}attention.self.query.bias"])
        self.k_proj.weight = nn.Parameter(state_dict[f"{prefix}attention.self.key.weight"])
        self.k_proj.bias = nn.Parameter(state_dict[f"{prefix}attention.self.key.bias"])
        self.v_proj.weight = nn.Parameter(state_dict[f"{prefix}attention.self.value.weight"])
        self.v_proj.bias = nn.Parameter(state_dict[f"{prefix}attention.self.value.bias"])
        self.attn_out_proj.weight = nn.Parameter(state_dict[f"{prefix}attention.output.dense.weight"])
        self.attn_out_proj.bias = nn.Parameter(state_dict[f"{prefix}attention.output.dense.bias"])

        # LayerNorm 权重
        self.attn_ln.weight = nn.Parameter(state_dict[f"{prefix}attention.output.LayerNorm.weight"])
        self.attn_ln.bias = nn.Parameter(state_dict[f"{prefix}attention.output.LayerNorm.bias"])
        self.ffn_ln.weight = nn.Parameter(state_dict[f"{prefix}output.LayerNorm.weight"])
        self.ffn_ln.bias = nn.Parameter(state_dict[f"{prefix}output.LayerNorm.bias"])

        # 前馈网络权重
        self.ffn_intermediate.weight = nn.Parameter(state_dict[f"{prefix}intermediate.dense.weight"])
        self.ffn_intermediate.bias = nn.Parameter(state_dict[f"{prefix}intermediate.dense.bias"])
        self.ffn_out.weight = nn.Parameter(state_dict[f"{prefix}output.dense.weight"])
        self.ffn_out.bias = nn.Parameter(state_dict[f"{prefix}output.dense.bias"])

    def forward(self, x):
        """x: [1, seq_len, 768]"""
        # 1. 自注意力层
        q = self.q_proj(x)  # [1, seq_len, 768]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 多头拆分
        q = self._split_heads(q)  # [1, 12, seq_len, 64]
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 注意力计算
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.attention_head_size, dtype=torch.float32))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # [1, 12, seq_len, 64]

        # 多头合并
        attn_out = self._merge_heads(attn_out)  # [1, seq_len, 768]
        attn_out = self.attn_out_proj(attn_out)  # [1, seq_len, 768]

        # 残差 + LayerNorm
        x = self.attn_ln(x + attn_out)  # [1, seq_len, 768]

        # 2. 前馈网络
        ffn_out = self.ffn_intermediate(x)  # [1, seq_len, 3072]
        ffn_out = F.gelu(ffn_out)  # PyTorch原生GELU
        ffn_out = self.ffn_out(ffn_out)  # [1, seq_len, 768]

        # 残差 + LayerNorm
        x = self.ffn_ln(x + ffn_out)  # [1, seq_len, 768]

        return x

    def _split_heads(self, x):
        """拆分多头：[1, seq_len, 768] -> [1, 12, seq_len, 64]"""
        batch_size, seq_len, hidden_size = x.shape
        return x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

    def _merge_heads(self, x):
        """合并多头：[1, 12, seq_len, 64] -> [1, seq_len, 768]"""
        batch_size, num_heads, seq_len, head_size = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

#PyTorchBert 输出（替换原DiyBert）
pt_bert = PyTorchBert(state_dict, num_layers=1)
pt_sequence_output, pt_pooler_output = pt_bert.forward(x)

#对比输出
print("官方 Bert：", ori_bert_sequence_output)
print("pytorch版Bert：", pt_sequence_output)
