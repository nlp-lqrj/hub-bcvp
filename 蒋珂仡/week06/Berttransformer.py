import torch
import torch.nn as nn
import math

class MyTransformerLayer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072):
        super(MyTransformerLayer, self).__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        # 1. Self-Attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        # 2. Feed Forward
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):

        residual = x
        q = self.transpose_for_scores(self.query(x))
        k = self.transpose_for_scores(self.key(x))
        v = self.transpose_for_scores(self.value(x))


        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        context_layer = torch.matmul(attention_probs, v)


        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)


        attention_output = self.attention_output(context_layer)


        hidden_states = self.layer_norm1(residual + attention_output)


        residual = hidden_states
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)


        output = self.layer_norm2(residual + layer_output)

        return output

def test_my_implementation():

    BATCH_SIZE = 2
    SEQ_LEN = 10
    HIDDEN_SIZE = 768
    HEADS = 12

    official_layer = nn.TransformerEncoderLayer(
        d_model=HIDDEN_SIZE,
        nhead=HEADS,
        dim_feedforward=3072,
        dropout=0.0,
        activation='gelu',
        batch_first=True,
        norm_first=False
    )


    my_layer = MyTransformerLayer(HIDDEN_SIZE, HEADS, 3072)

    with torch.no_grad():

        qkv_w = official_layer.self_attn.in_proj_weight.chunk(3, dim=0)
        qkv_b = official_layer.self_attn.in_proj_bias.chunk(3, dim=0)

        my_layer.query.weight.copy_(qkv_w[0])
        my_layer.query.bias.copy_(qkv_b[0])
        my_layer.key.weight.copy_(qkv_w[1])
        my_layer.key.bias.copy_(qkv_b[1])
        my_layer.value.weight.copy_(qkv_w[2])
        my_layer.value.bias.copy_(qkv_b[2])


        my_layer.attention_output.weight.copy_(official_layer.self_attn.out_proj.weight)
        my_layer.attention_output.bias.copy_(official_layer.self_attn.out_proj.bias)


        my_layer.layer_norm1.weight.copy_(official_layer.norm1.weight)
        my_layer.layer_norm1.bias.copy_(official_layer.norm1.bias)


        my_layer.intermediate.weight.copy_(official_layer.linear1.weight)
        my_layer.intermediate.bias.copy_(official_layer.linear1.bias)


        my_layer.output.weight.copy_(official_layer.linear2.weight)
        my_layer.output.bias.copy_(official_layer.linear2.bias)


        my_layer.layer_norm2.weight.copy_(official_layer.norm2.weight)
        my_layer.layer_norm2.bias.copy_(official_layer.norm2.bias)


    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)


    official_layer.eval()
    my_layer.eval()

    with torch.no_grad():
        official_output = official_layer(x)
        my_output = my_layer(x)


    diff = torch.max(torch.abs(official_output - my_output))

    print("\n---------------- 测试 ----------------")
    print(f"输入形状: {x.shape}")
    print(f"官方输出形状: {official_output.shape}")
    print(f"手写输出形状: {my_output.shape}")
    print(f"最大误差值: {diff.item():.8f}")

    if diff < 1e-5:
        print("\n代码逻辑与 PyTorch 官方实现一致！")
    else:
        print("\n 验证失败，误差过大，请检查代码逻辑。")


if __name__ == "__main__":
    test_my_implementation()
