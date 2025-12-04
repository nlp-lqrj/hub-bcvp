import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        """
        单个Transformer层 - 简化实现

        参数:
            d_model: 特征维度
            nhead: 注意力头的数量
            dim_feedforward: 前馈网络的维度
            dropout: dropout概率
        """
        super(TransformerLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        # 确保d_model可以被nhead整除
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        self.d_k = d_model // nhead  # 每个头的维度

        # 自注意力的Q、K、V投影矩阵
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力计算过程
        """
        # 计算Q和K的点积
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 缩放
        scores = scores / math.sqrt(self.d_k)

        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 应用dropout到注意力权重
        attn_weights = self.dropout(attn_weights)

        # 乘以V得到输出
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

    def multi_head_attention(self, x, mask=None):
        """
        多头注意力计算过程
        """
        batch_size, seq_len, d_model = x.size()

        # 线性投影得到Q、K、V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # 计算缩放点积注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        # 输出投影
        output = self.W_o(attn_output)

        return output, attn_weights

    def forward(self, x, mask=None):
        """
        前向传播
        """
        # 第一个层归一化 (Pre-LN)
        x_norm1 = self.norm1(x)

        # 多头自注意力
        attn_output, attn_weights = self.multi_head_attention(x_norm1, mask)

        # 注意力dropout和残差连接
        x = x + self.dropout(attn_output)

        # 第二个层归一化 (Pre-LN)
        x_norm2 = self.norm2(x)

        # 前馈网络
        ff_output = self.ffn(x_norm2)

        # 前馈网络dropout和残差连接
        x = x + self.dropout(ff_output)

        return x, attn_weights


# 计算展示
if __name__ == "__main__":
    # 创建模型
    model = TransformerLayer(d_model=785, nhead=5)

    # 输入数据
    batch_size = 2
    seq_len = 666
    input_tensor = torch.randn(batch_size, seq_len, 785)

    # 创建因果掩码（可选）
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

    # 前向传播
    output, attn_weights = model(input_tensor, mask)

    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
