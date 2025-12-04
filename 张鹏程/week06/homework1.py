import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# 为了结果可复现，设置随机种子
torch.manual_seed(42)

# 1. 加载预训练模型和权重
bert = BertModel.from_pretrained(r"bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()

# 2. 定义输入（模拟一个4个字的句子）
x = torch.LongTensor([2450, 15486, 102, 2110])


# 3. 定义DIY Bert模型 (PyTorch版)
class DiyBert(nn.Module):
    def __init__(self, state_dict):
        super(DiyBert, self).__init__()  # 继承nn.Module的构造函数

        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 6  # 注意：这里的层数要和预训练模型的配置一致

        # 4. 加载权重
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        # Embedding部分
        # 使用nn.Embedding层来替代手动的索引操作
        self.word_embeddings = nn.Embedding.from_pretrained(state_dict["embeddings.word_embeddings.weight"])
        self.position_embeddings = nn.Embedding.from_pretrained(state_dict["embeddings.position_embeddings.weight"])
        self.token_type_embeddings = nn.Embedding.from_pretrained(state_dict["embeddings.token_type_embeddings.weight"])

        # 冻结embedding层（如果只是做特征提取）
        self.word_embeddings.requires_grad_(False)
        self.position_embeddings.requires_grad_(False)
        self.token_type_embeddings.requires_grad_(False)

        # LayerNorm层
        self.embeddings_layer_norm = nn.LayerNorm(self.hidden_size)
        self.embeddings_layer_norm.weight.data = state_dict["embeddings.LayerNorm.weight"]
        self.embeddings_layer_norm.bias.data = state_dict["embeddings.LayerNorm.bias"]

        # Transformer部分
        self.transformer_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer_layers.append(
                BertLayer(state_dict, i, self.num_attention_heads, self.hidden_size)
            )

        # Pooler层
        self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
        self.pooler.weight.data = state_dict["pooler.dense.weight"]
        self.pooler.bias.data = state_dict["pooler.dense.bias"]

    def forward(self, x):
        # x.shape = [max_len]

        # Embedding层 forward
        we = self.word_embeddings(x)  # shape: [max_len, hidden_size]

        # position embeding的输入 [0, 1, 2, 3]
        position_ids = torch.arange(len(x), dtype=torch.long, device=x.device)
        pe = self.position_embeddings(position_ids)  # shape: [max_len, hidden_size]

        # token type embedding, 单输入的情况下为[0, 0, 0, 0]
        token_type_ids = torch.zeros_like(x)
        te = self.token_type_embeddings(token_type_ids)  # shape: [max_len, hidden_size]

        embedding = we + pe + te
        embedding = self.embeddings_layer_norm(embedding)  # shape: [max_len, hidden_size]

        # Transformer层 forward
        sequence_output = embedding
        for layer in self.transformer_layers:
            sequence_output = layer(sequence_output)

        # Pooler层 forward
        # BERT使用第一个token ([CLS])的输出作为句子的整体表示
        pooler_output = self.pooler(sequence_output[0])
        pooler_output = torch.tanh(pooler_output)

        return sequence_output, pooler_output


# 5. 定义单个Transformer层 (PyTorch版)
class BertLayer(nn.Module):
    def __init__(self, state_dict, layer_index, num_attention_heads, hidden_size):
        super(BertLayer, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = int(hidden_size / num_attention_heads)

        # Attention部分
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.query.weight.data = state_dict[f"encoder.layer.{layer_index}.attention.self.query.weight"]
        self.query.bias.data = state_dict[f"encoder.layer.{layer_index}.attention.self.query.bias"]
        self.key.weight.data = state_dict[f"encoder.layer.{layer_index}.attention.self.key.weight"]
        self.key.bias.data = state_dict[f"encoder.layer.{layer_index}.attention.self.key.bias"]
        self.value.weight.data = state_dict[f"encoder.layer.{layer_index}.attention.self.value.weight"]
        self.value.bias.data = state_dict[f"encoder.layer.{layer_index}.attention.self.value.bias"]

        # Attention输出投影
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_output.weight.data = state_dict[f"encoder.layer.{layer_index}.attention.output.dense.weight"]
        self.attention_output.bias.data = state_dict[f"encoder.layer.{layer_index}.attention.output.dense.bias"]

        # Attention后的LayerNorm
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.attention_layer_norm.weight.data = state_dict[
            f"encoder.layer.{layer_index}.attention.output.LayerNorm.weight"]
        self.attention_layer_norm.bias.data = state_dict[f"encoder.layer.{layer_index}.attention.output.LayerNorm.bias"]

        # Feed Forward部分
        self.intermediate = nn.Linear(hidden_size, 4 * hidden_size)  # BERT的中间层维度通常是hidden_size的4倍
        self.intermediate.weight.data = state_dict[f"encoder.layer.{layer_index}.intermediate.dense.weight"]
        self.intermediate.bias.data = state_dict[f"encoder.layer.{layer_index}.intermediate.dense.bias"]

        self.output = nn.Linear(4 * hidden_size, hidden_size)
        self.output.weight.data = state_dict[f"encoder.layer.{layer_index}.output.dense.weight"]
        self.output.bias.data = state_dict[f"encoder.layer.{layer_index}.output.dense.bias"]

        # Feed Forward后的LayerNorm
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
        self.ff_layer_norm.weight.data = state_dict[f"encoder.layer.{layer_index}.output.LayerNorm.weight"]
        self.ff_layer_norm.bias.data = state_dict[f"encoder.layer.{layer_index}.output.LayerNorm.bias"]

    def forward(self, x):
        # x.shape = [max_len, hidden_size]

        # Self-Attention
        attn_output = self.self_attention(x)

        # 残差连接 + LayerNorm
        x = self.attention_layer_norm(x + attn_output)

        # Feed Forward
        ff_output = self.feed_forward(x)

        # 残差连接 + LayerNorm
        x = self.ff_layer_norm(x + ff_output)

        return x

    def self_attention(self, x):
        # x.shape = [max_len, hidden_size]

        # 1. 线性投影得到Q, K, V
        q = self.query(x)  # [max_len, hidden_size]
        k = self.key(x)  # [max_len, hidden_size]
        v = self.value(x)  # [max_len, hidden_size]

        # 2. 多头分割
        batch_size = 1  # 因为我们的输入是单个序列
        q = self.transpose_for_scores(q)  # [num_heads, max_len, head_size]
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # 3. 计算注意力分数
        # (num_heads, max_len, head_size) @ (num_heads, head_size, max_len) -> (num_heads, max_len, max_len)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))

        # 4. 计算注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)  # [num_heads, max_len, max_len]

        # 5. 应用注意力到V上
        # (num_heads, max_len, max_len) @ (num_heads, max_len, head_size) -> (num_heads, max_len, head_size)
        context_layer = torch.matmul(attention_probs, v)

        # 6. 多头合并
        context_layer = context_layer.permute(1, 0, 2).contiguous()  # [max_len, num_heads, head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [max_len, hidden_size]

        # 7. 最终线性投影
        attention_output = self.attention_output(context_layer)  # [max_len, hidden_size]

        return attention_output

    def transpose_for_scores(self, x):
        # x.shape = [max_len, hidden_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # [max_len, num_heads, head_size]
        return x.permute(1, 0, 2)  # [num_heads, max_len, head_size]

    def feed_forward(self, x):
        # x.shape = [max_len, hidden_size]

        # 1. 中间层
        x = self.intermediate(x)  # [max_len, 4*hidden_size]
        x = F.gelu(x)

        # 2. 输出层
        x = self.output(x)  # [max_len, hidden_size]

        return x


# --- 运行和验证 ---

# 实例化我们的PyTorch DIY Bert模型
db_pytorch = DiyBert(state_dict)

# 设置为评估模式
db_pytorch.eval()

# 执行前向传播
# 注意：在PyTorch中，如果模型包含dropout或batchnorm等层，最好使用with torch.no_grad():来禁用梯度计算，节省资源
with torch.no_grad():
    pytorch_sequence_output, pytorch_pooler_output = db_pytorch(x)

print("PyTorch 实现 - Sequence Output Shape:", pytorch_sequence_output.shape)
print("PyTorch 实现 - Pooler Output Shape:", pytorch_pooler_output.shape)
print("\nPyTorch 实现 - Pooler Output:")
print(pytorch_pooler_output)

# 为了验证正确性，我们可以与原始的Hugging Face BertModel输出进行比较
print("\n" + "=" * 50 + "\n")
print("Hugging Face BertModel 输出对比:")
bert.eval()
with torch.no_grad():
    hf_sequence_output, hf_pooler_output = bert(x.unsqueeze(0))  # Hugging Face的输入需要 batch维度

print("Hugging Face - Sequence Output Shape:", hf_sequence_output.shape)
print("Hugging Face - Pooler Output Shape:", hf_pooler_output.shape)
print("\nHugging Face - Pooler Output:")
print(hf_pooler_output)

# 检查两个模型的输出是否接近（应该非常接近，因为我们使用了相同的权重）
print("\n" + "=" * 50 + "\n")
print("输出差异 (L2 Norm):", torch.norm(pytorch_pooler_output - hf_pooler_output.squeeze()).item())
