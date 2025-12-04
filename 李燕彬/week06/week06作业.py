import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力机制模块"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        参数说明:
            d_model: 模型的维度（词嵌入维度）
            n_heads: 注意力头的数量（需满足 d_model % n_heads == 0）
            dropout: dropout概率
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个注意力头的维度

        # 定义Q、K、V的线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出线性变换层
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))  # 缩放因子（防止点积过大）

    def forward(self, q, k, v, mask=None):
        """
        前向传播:
            q: 查询向量 [batch_size, seq_len_q, d_model]
            k: 键向量 [batch_size, seq_len_k, d_model]
            v: 值向量 [batch_size, seq_len_v, d_model]（通常seq_len_k == seq_len_v）
            mask: 掩码矩阵 [batch_size, n_heads, seq_len_q, seq_len_k] 或 None
        返回:
            output: 多头注意力输出 [batch_size, seq_len_q, d_model]
            attention: 注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)

        # 1. 线性变换并拆分多头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                               2)  # [batch_size, n_heads, seq_len_q, d_k]
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                               2)  # [batch_size, n_heads, seq_len_k, d_k]
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                               2)  # [batch_size, n_heads, seq_len_v, d_k]

        # 2. 计算注意力分数（Q·K^T / √d_k）
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale.to(
            q.device)  # [batch_size, n_heads, seq_len_q, seq_len_k]

        # 3. 应用掩码（若有）：将掩码位置设为-1e10，softmax后趋近于0
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)

        # 4. 计算注意力权重（softmax）并dropout
        attention = F.softmax(attention_scores, dim=-1)  # [batch_size, n_heads, seq_len_q, seq_len_k]
        attention = self.dropout(attention)

        # 5. 加权求和（注意力权重 × V）
        x = torch.matmul(attention, v)  # [batch_size, n_heads, seq_len_q, d_k]

        # 6. 拼接多头并线性变换
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len_q, d_model]
        output = self.w_o(x)

        return output, attention


class PositionwiseFeedforward(nn.Module):
    """位置前馈网络模块"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        参数说明:
            d_model: 模型维度
            d_ff: 隐藏层维度（通常d_ff > d_model）
            dropout: dropout概率
        """
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播:
            x: 输入 [batch_size, seq_len, d_model]
        返回:
            输出 [batch_size, seq_len, d_model]
        """
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    """Transformer Encoder层（单个）"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        参数说明:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        super(TransformerLayer, self).__init__()
        # 多头注意力层
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈网络层
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播（Encoder层流程：Self-Attention → 残差+LN → FFN → 残差+LN）:
            x: 输入序列 [batch_size, seq_len, d_model]
            mask: 自注意力掩码 [batch_size, 1, seq_len, seq_len] 或 None
        返回:
            out: 输出序列 [batch_size, seq_len, d_model]
            attention: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        # 1. 多头自注意力 + 残差连接 + 层归一化
        attn_output, attention = self.self_attn(x, x, x, mask)  # 自注意力：Q=K=V=x
        x = self.norm1(x + self.dropout1(attn_output))  # 残差连接 + LN

        # 2. 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        out = self.norm2(x + self.dropout2(ff_output))  # 残差连接 + LN

        return out, attention


# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 超参数设置
    batch_size = 2
    seq_len = 5
    d_model = 128  # 模型维度
    n_heads = 8  # 注意力头数（128/8=16，每个头维度16）
    d_ff = 512  # 前馈网络隐藏层维度
    dropout = 0.1

    # 创建Transformer层实例
    transformer_layer = TransformerLayer(d_model, n_heads, d_ff, dropout)

    # 生成随机输入（batch_size, seq_len, d_model）
    x = torch.randn(batch_size, seq_len, d_model)

    # 生成掩码（例如：padding mask，假设seq_len=5，前3个token有效，后2个padding）
    mask = torch.ones(batch_size, 1, seq_len, seq_len)  # [batch_size, 1, seq_len, seq_len]
    mask[:, :, :, 3:] = 0  # 禁止关注padding部分

    # 前向传播
    output, attention = transformer_layer(x, mask)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 应与输入形状一致：[2, 5, 128]
    print(f"注意力权重形状: {attention.shape}")  # [2, 8, 5, 5]（batch_size, n_heads, seq_len, seq_len）