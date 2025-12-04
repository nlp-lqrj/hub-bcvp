import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 位置编码
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 非参数，但随模型保存/载入

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

# ---------------------------
# 多头自注意力（batch_first）
# ---------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query/key/value: (batch, seq_len, d_model)
        B, Tq, _ = query.size()
        _, Tk, _ = key.size()

        q = self.q_proj(query)  # (B, Tq, d_model)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 分头: (B, num_heads, seq_len, head_dim)
        def shape(x):
            return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q = shape(q)
        k = shape(k)
        v = shape(v)

        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, Tq, Tk)

        if mask is not None:
            # mask should be broadcastable to (B, num_heads, Tq, Tk)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (B, nh, Tq, head_dim)

        # 合并头: (B, Tq, d_model)
        context = context.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        out = self.out_proj(context)
        return out, attn  # 返回注意力权重以便调试/可视化

# ---------------------------
# 前馈层
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# 编码器层
# ---------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 自注意力
        attn_out, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

# ---------------------------
# 解码器层
# ---------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        # tgt 自注意力（带未来位屏蔽）
        self_attn_out, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # 与 encoder 输出做交叉注意力
        cross_attn_out, _ = self.cross_attn(x, enc_out, enc_out, mask=memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return x

# ---------------------------
# 编码器 & 解码器 堆叠
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src_tokens, src_mask=None):
        # src_tokens: (B, src_len)
        x = self.tok_embed(src_tokens) * math.sqrt(self.tok_embed.embedding_dim)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return self.norm(x)  # (B, src_len, d_model)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt_tokens, enc_out, tgt_mask=None, memory_mask=None):
        x = self.tok_embed(tgt_tokens) * math.sqrt(self.tok_embed.embedding_dim)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(x)  # (B, tgt_len, d_model)

# ---------------------------
# 完整 Transformer
# ---------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6, num_heads=8, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, N, num_heads, d_ff, max_len, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab)  # output projection to vocab

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None, memory_mask=None):
        # src_tokens: (B, src_len), tgt_tokens: (B, tgt_len)
        enc_out = self.encoder(src_tokens, src_mask)
        dec_out = self.decoder(tgt_tokens, enc_out, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.generator(dec_out)  # (B, tgt_len, tgt_vocab)
        return logits

# ---------------------------
# Mask helpers
# ---------------------------
def create_padding_mask(seq, pad_token=0):
    # seq: (B, seq_len) -> mask: (B, 1, 1, seq_len) or (B, 1, seq_len) depending needs
    # We'll create a mask with shape (B, 1, 1, seq_len) for broadcasting to (B, nh, Tq, Tk)
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(1)  # True where not padding
    # Make it int (1 for valid, 0 for pad) to use with masked_fill
    return mask  # bool mask

def create_look_ahead_mask(sz):
    # returns (1, 1, sz, sz) or (sz, sz) mask for future positions (upper triangular)
    mask = torch.tril(torch.ones((sz, sz), dtype=torch.uint8))
    return mask.unsqueeze(0).unsqueeze(0)  # shape (1,1,sz,sz)

def combine_masks(pad_mask_src, pad_mask_tgt, tgt_len):
    # pad_mask_src: (B,1,1,src_len); pad_mask_tgt: (B,1,1,tgt_len)
    # create look-ahead mask for tgt and combine with tgt padding mask
    look = create_look_ahead_mask(tgt_len).to(pad_mask_tgt.device)  # (1,1,tgt_len,tgt_len)
    tgt_mask = pad_mask_tgt & look  # broadcast & (B,1,tgt_len,tgt_len)
    return tgt_mask

# ---------------------------
# 简单使用示例
# ---------------------------
if __name__ == "__main__":
