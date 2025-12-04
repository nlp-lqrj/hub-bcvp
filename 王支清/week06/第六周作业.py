import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask（如果有）
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到value
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 最终线性变换
        output = self.w_o(attn_output)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 或者使用 nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层 + 残差连接 + 层归一化
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络子层 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # 自注意力子层（掩码的）
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # 交叉注意力子层
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x, memory, memory, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x, self_attn_weights, cross_attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_length, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 嵌入 + 位置编码
        x = self.token_embedding(src)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            all_attn_weights.append(attn_weights)

        return x, all_attn_weights


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_length, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        # 嵌入 + 位置编码
        x = self.token_embedding(tgt)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        all_self_attn_weights = []
        all_cross_attn_weights = []

        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, memory, src_mask, tgt_mask
            )
            all_self_attn_weights.append(self_attn_weights)
            all_cross_attn_weights.append(cross_attn_weights)

        output = self.linear(x)
        return output, all_self_attn_weights, all_cross_attn_weights


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 d_ff=2048, num_layers=6, max_seq_length=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, d_ff, num_layers,
            max_seq_length, dropout
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, d_ff, num_layers,
            max_seq_length, dropout
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory, enc_attn_weights = self.encoder(src, src_mask)
        output, dec_self_attn_weights, dec_cross_attn_weights = self.decoder(
            tgt, memory, src_mask, tgt_mask
        )
        return output, enc_attn_weights, dec_self_attn_weights, dec_cross_attn_weights


# 使用示例
if __name__ == "__main__":
    # 超参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    max_seq_length = 100
    batch_size = 32
    seq_len = 50

    # 创建模型
    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads,
        d_ff, num_layers, max_seq_length
    )

    # 创建示例数据
    src = torch.randint(0, src_vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))

    # 前向传播
    output, enc_attn, dec_self_attn, dec_cross_attn = transformer(src, tgt)

    print(f"输入形状: {src.shape}")
    print(f"输出形状: {output.shape}")
    print(f"编码器注意力权重数量: {len(enc_attn)}")
    print(f"解码器自注意力权重数量: {len(dec_self_attn)}")
    print(f"解码器交叉注意力权重数量: {len(dec_cross_attn)}")
