import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleTransformerLayer(nn.Module):
    """单层Transformer实现，对应BERT的一个编码器层"""

    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072, dropout_prob=0.1):
        super(SingleTransformerLayer, self).__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # 自注意力层
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_attention_heads, dropout_prob)

        # 第一个LayerNorm和残差连接（在注意力之后）
        self.attention_layer_norm = nn.LayerNorm(hidden_size)

        # 前馈神经网络
        self.feed_forward = FeedForwardNetwork(hidden_size, intermediate_size, dropout_prob)

        # 第二个LayerNorm和残差连接（在前馈网络之后）
        self.output_layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] 可选
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        # 自注意力 + 残差连接 + LayerNorm
        attention_output = self.self_attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        # 残差连接和LayerNorm
        hidden_states = self.attention_layer_norm(hidden_states + attention_output)

        # 前馈网络 + 残差连接 + LayerNorm
        feed_forward_output = self.feed_forward(hidden_states)
        feed_forward_output = self.dropout(feed_forward_output)
        # 残差连接和LayerNorm
        hidden_states = self.output_layer_norm(hidden_states + feed_forward_output)

        return hidden_states


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, hidden_size=768, num_attention_heads=12, dropout_prob=0.1):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Q, K, V 线性变换
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # 输出线性层
        self.output_dense = nn.Linear(self.all_head_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        """将输入重排为多头格式"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_size]

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] 可选
        """
        batch_size, seq_len = hidden_states.size()[:2]

        # 计算Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))  # [batch, num_heads, seq_len, head_size]
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 计算注意力分数: Q * K^T / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用注意力掩码（如果有）
        if attention_mask is not None:
            # 将注意力掩码扩展到与注意力分数相同的形状
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_scores = attention_scores + (attention_mask * -10000.0)

        # Softmax归一化
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 注意力加权: 注意力概率 * V
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch, num_heads, seq_len, head_size]

        # 合并多头
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, num_heads, head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [batch, seq_len, hidden_size]

        # 输出投影
        attention_output = self.output_dense(context_layer)

        return attention_output


class FeedForwardNetwork(nn.Module):
    """前馈神经网络（两层全连接层）"""

    def __init__(self, hidden_size=768, intermediate_size=3072, dropout_prob=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states):
        # 第一层：扩展到更大维度 + GELU激活
        intermediate_output = self.intermediate_dense(hidden_states)
        intermediate_output = F.gelu(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)

        # 第二层：投影回原始维度
        output = self.output_dense(intermediate_output)

        return output


# 使用示例
if __name__ == "__main__":
    # 参数设置（与BERT-base一致）
    hidden_size = 768
    num_attention_heads = 12
    intermediate_size = 3072
    dropout_prob = 0.1

    # 创建单层Transformer
    transformer_layer = SingleTransformerLayer(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        dropout_prob=dropout_prob
    )

    # 模拟输入（与原始代码相同的输入）
    # batch_size=1, seq_len=4, hidden_size=768
    batch_size = 1
    seq_len = 4
    example_input = torch.randn(batch_size, seq_len, hidden_size)

    print(f"输入形状: {example_input.shape}")

    # 前向传播
    output = transformer_layer(example_input)

    print(f"输出形状: {output.shape}")
    print(f"输入输出形状是否一致: {example_input.shape == output.shape}")

    # 可以创建多个层来构建完整的Transformer编码器
    num_layers = 6
    transformer_layers = nn.ModuleList([
        SingleTransformerLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout_prob=dropout_prob
        ) for _ in range(num_layers)
    ])

    print(f"\n创建了{num_layers}层Transformer编码器")
