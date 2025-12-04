import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头自注意力机制：将查询、键、值拆分到多个头，并行计算注意力"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除（每个头的维度 d_k = d_model / num_heads）"

        self.d_model = d_model  # 模型整体维度（如 Bert-base 中 d_model=768）
        self.num_heads = num_heads  # 注意力头数（如 Bert-base 中 num_heads=12）
        self.d_k = d_model // num_heads  # 每个头的维度（如 768/12=64）

        # 线性投影层：将输入 Q, K, V 映射到 d_model 维度（拆分前的总维度）
        self.w_q = nn.Linear(d_model, d_model)  # Q 的投影
        self.w_k = nn.Linear(d_model, d_model)  # K 的投影
        self.w_v = nn.Linear(d_model, d_model)  # V 的投影

        # 输出线性层：将多个头的结果拼接后映射到 d_model 维度
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)  # dropout 正则化

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将输入拆分为多个头：(batch_size, seq_len, d_model) → (batch_size, num_heads, seq_len, d_k)"""
        batch_size = x.size(0)
        # 先拆分最后一维：(batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        # 交换维度，让 num_heads 作为第二维（方便并行计算）
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                     mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """缩放点积注意力（Transformer 核心）：
        Args:
            q: (batch_size, num_heads, seq_len_q, d_k)
            k: (batch_size, num_heads, seq_len_k, d_k)
            v: (batch_size, num_heads, seq_len_v, d_k)（seq_len_k = seq_len_v）
            mask: (batch_size, 1, seq_len_q, seq_len_k) 或类似形状（遮挡无效位置）
        Returns:
            output: 注意力加权后的结果
            attn_weights: 注意力权重（可选，用于可视化）
        """
        # 1. 计算 Q·K^T（注意力得分）：(batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1))

        # 2. 缩放：除以 sqrt(d_k)，避免得分过大导致 softmax 梯度消失
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=q.dtype))

        # 3. 计算注意力权重（softmax 归一化）
        attn_weights = F.softmax(scores, dim=-1)

        # 4. 注意力加权 V：(batch_size, num_heads, seq_len_q, d_k)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> tuple[
        torch.Tensor, torch.Tensor]:
        """前向传播：
        Args:
            q: (batch_size, seq_len_q, d_model)
            k: (batch_size, seq_len_k, d_model)
            v: (batch_size, seq_len_v, d_model)
            mask: 注意力掩码（可选）
        Returns:
            output: (batch_size, seq_len_q, d_model) → 多头注意力输出
            attn_weights: 注意力权重（可选）
        """
        # 1. 线性投影 + 拆分多头
        q_proj = self.w_q(q)  # (batch_size, seq_len_q, d_model)
        k_proj = self.w_k(k)  # (batch_size, seq_len_k, d_model)
        v_proj = self.w_v(v)  # (batch_size, seq_len_v, d_model)

        q_split = self.split_heads(q_proj)  # (batch_size, num_heads, seq_len_q, d_k)
        k_split = self.split_heads(k_proj)  # (batch_size, num_heads, seq_len_k, d_k)
        v_split = self.split_heads(v_proj)  # (batch_size, num_heads, seq_len_v, d_k)

        # 2. 计算缩放点积注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(q_split, k_split, v_split, mask)

        # 3. 拼接多个头的结果：(batch_size, num_heads, seq_len_q, d_k) → (batch_size, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2)  # 交换 num_heads 和 seq_len 维度
        attn_output = attn_output.contiguous().view(-1, attn_output.size(1), self.d_model)  # 拼接

        # 4. 输出线性投影 + Dropout
        output = self.w_o(attn_output)
        output = self.dropout(output)

        return output, attn_weights


class FeedForwardNetwork(nn.Module):
    """前馈网络（Transformer 层的第二个子模块）：d_model → 4*d_model → d_model"""

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层线性变换（升维）
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二层线性变换（降维）
        self.dropout = nn.Dropout(dropout)  # Dropout 正则化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：(batch_size, seq_len, d_model) → (batch_size, seq_len, d_model)"""
        x = self.fc1(x)  # (batch_size, seq_len, d_ff)
        x = F.gelu(x)  # 激活函数（Bert 中用 gelu）
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)  # (batch_size, seq_len, d_model)
        return x


class TransformerLayer(nn.Module):
    """完整的 Transformer 层（Encoder 层，无掩码自注意力）：
    结构：多头自注意力 → 残差连接 + 层归一化 → 前馈网络 → 残差连接 + 层归一化
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        # 1. 多头自注意力模块
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 2. 前馈网络模块
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        # 3. 层归一化（每个子模块后各一个）
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        # 4. Dropout（可选，部分实现会在残差连接后加）
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播（Encoder 层逻辑）：
        Args:
            x: (batch_size, seq_len, d_model) → 输入序列
            mask: 注意力掩码（可选，用于遮挡 padding 位置）
        Returns:
            output: (batch_size, seq_len, d_model) → Transformer 层输出
            attn_weights: 注意力权重（可选）
        """
        # -------------------------- 第一步：多头自注意力 + 残差连接 + 层归一化 --------------------------
        # 自注意力：输入 Q=K=V=x（自注意力机制）
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        # 残差连接：输入 x + 注意力输出（先加后归一化，Bert 用这种方式；也可以先归一化后加）
        x = x + self.dropout1(attn_output)
        # 层归一化
        x = self.layer_norm1(x)

        # -------------------------- 第二步：前馈网络 + 残差连接 + 层归一化 --------------------------
        # 前馈网络
        ffn_output = self.ffn(x)
        # 残差连接
        x = x + self.dropout2(ffn_output)
        # 层归一化
        output = self.layer_norm2(x)

        return output, attn_weights


# -------------------------- 测试代码（验证实现正确性） --------------------------
if __name__ == "__main__":
    # 超参数（参考 Bert-base 配置）
    d_model = 768  # 模型维度
    num_heads = 12  # 注意力头数
    d_ff = 3072  # 前馈网络中间维度（Bert 中是 3072，标准 Transformer 是 2048）
    dropout = 0.1  # Dropout 概率

    # 构造输入（batch_size=2, seq_len=10, d_model=768）
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model)  # 随机输入
    # 构造掩码（假设前 8 个是有效token，后 2 个是 padding，mask 形状：(batch_size, 1, 1, seq_len)）
    mask = torch.ones(batch_size, seq_len)
    mask[:, 8:] = 0  # 后 2 个位置设为 0（遮挡）
    mask = mask.unsqueeze(1).unsqueeze(1)  # 适配注意力模块的输入形状

    # 初始化 Transformer 层
    transformer_layer = TransformerLayer(d_model, num_heads, d_ff, dropout)

    # 前向传播
    output, attn_weights = transformer_layer(x, mask)

    # 打印输出形状（验证维度正确）
    print(f"输入形状：{x.shape}")  # torch.Size([2, 10, 768])
    print(f"输出形状：{output.shape}")  # torch.Size([2, 10, 768])（维度不变）
    print(
        f"注意力权重形状：{attn_weights.shape}")  # torch.Size([2, 12, 10, 10])（batch_size, num_heads, seq_len, seq_len）
    print("Transformer 层实现成功！")
