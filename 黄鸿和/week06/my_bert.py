import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel as OfficialBertModel

# ------ Bert嵌入层 ------
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_model, max_length, split_sen_num):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, hidden_model)
        self.split_embeddings = nn.Embedding(split_sen_num, hidden_model)
        self.position_embeddings = nn.Embedding(max_length, hidden_model)

        self.layernorm = nn.LayerNorm(hidden_model)
    
    def forward(self, input_ids, split_type):
        seq_length = input_ids.size(1)
        # [b,seq_length]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_embedding = self.position_embeddings(position_ids)

        word_embedding = self.word_embeddings(input_ids)
        
        if split_type is None:
            split_type = torch.zeros_like(input_ids)
        split_embedding = self.split_embeddings(split_type)

        embeddings = position_embedding + word_embedding + split_embedding

        embeddings = self.layernorm(embeddings)

        return embeddings

# ------ 多头注意力层 ------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_model, nums_head):
        super().__init__()
        self.h_model = hidden_model
        self.nums_head = nums_head
        self.head_dim = hidden_model // nums_head

        self.q_linear = nn.Linear(hidden_model, hidden_model)
        self.k_linear = nn.Linear(hidden_model, hidden_model)
        self.v_linear = nn.Linear(hidden_model, hidden_model)
        self.cat_output_linear = nn.Linear(hidden_model, hidden_model)
        
        assert hidden_model % nums_head == 0, "隐藏层维度不能被头数整除"

    def forward(self, q, k, v):
        # q -> (B, L, hidden_model)
        bs = q.size(0)
        # Q -> (B, L, hidden_model) -> (B, L, H * head_dim) -> (B, L, H, head_dim) -> (B, H, L, head_dim)
        # 把头的维度放在前面
        Q = self.q_linear(q).view(bs, -1, self.nums_head, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(bs, -1, self.nums_head, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(bs, -1, self.nums_head, self.head_dim).transpose(1, 2)
        # weight_scores -> [B, H, L, L]
        # 注意力计算：Q * K^T / sqrt(d_k)
        weight_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # 进行权重数值归一化
        att_weight = F.softmax(weight_scores, dim=-1)

        # 与 V 进行加权计算
        # context -> (B, H, L, head_dim)
        context = torch.matmul(att_weight, V) # Bug fix: Should use V, not v
        # context -> (B, H, L, head_dim) -> (B, L, hidden_model)
        # 确保张量（Tensor）在内存中是连续存储的
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.h_model)

        context = self.cat_output_linear(context)
        return context

# -------- FFN --------
class FFN(nn.Module):
    def __init__(self, hidden_model, hidden_model_1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_model, hidden_model_1)
        self.linear2 = nn.Linear(hidden_model_1, hidden_model)
        self.gelu = nn.GELU()
    def forward(self, x):
        x = self.gelu(self.linear1(x))
        x = self.linear2(x)
        return x

#   -------- Encoder layers --------
class Encoder(nn.Module):
    def __init__(self, hidden_model, hidden_model_1, nums_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_model)
        self.self_atten = MultiHeadAttention(hidden_model, nums_head)
        self.norm2 = nn.LayerNorm(hidden_model)
        self.ffn = FFN(hidden_model, hidden_model_1)
        
    def forward(self, x):
        # Self-Attention 子层
        x_atten = self.self_atten(x, x, x)
        # 残差连接： x_new = x + x_atten
        x = x + x_atten
        x = self.norm1(x) 

        # FFN 子层
        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.norm2(x)

        return x
    
# 组合成为 Bert模型
class BertModel(nn.Module):
    # 为避免混淆，使用默认参数
    def __init__(self, vocab_size, hidden_model=768, hidden_model_1 = 768 * 4, nums_head = 12, max_length = 512, split_sen_num = 2, nums_encoder_layer = 12):
        super().__init__()

        self.embeddings = BertEmbeddings(vocab_size, hidden_model, max_length, split_sen_num)
        self.encoder_layers = nn.ModuleList([
            Encoder(hidden_model, hidden_model_1, nums_head) for _ in range(nums_encoder_layer)
        ])

        self.pooler = nn.Sequential(
            nn.Linear(hidden_model, hidden_model),
            nn.Tanh()
        )
    def forward(self, input_ids, split_type=None):
        embedding_output = self.embeddings(input_ids, split_type)
        encoder_output = embedding_output

        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        
		# 仅取第一个 token (通常是 [CLS]) 的输出来进行池化
        first_token_tensor = encoder_output[:, 0]
        encoder_output_pooler = self.pooler(first_token_tensor)
        
        # 返回所有 token 的输出和 [CLS] token 的池化输出
        return encoder_output, encoder_output_pooler
    
if __name__ == '__main__':
    # 参数
    vocab_size = 21128
    # 隐藏层维度，通常为 768
    hidden_model_size = 768 
    # FFN 中间层维度，通常是 hidden_model_size * 4
    hidden_model_ffn = 768 * 4
    nums_head = 12
    max_length = 512
    split_sen_num = 2 # 句子A和句子B (0和1)
    nums_encoder_layer = 12
    

    model = BertModel(
        vocab_size=vocab_size, 
        hidden_model=hidden_model_size, 
        hidden_model_1=hidden_model_ffn, 
        nums_head=nums_head, 
        max_length=max_length, 
        split_sen_num=split_sen_num, 
        nums_encoder_layer=nums_encoder_layer
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    bs = 1
    seq_length = 50
    # 模拟输入
    input_ids = torch.randint(0, vocab_size, (bs, seq_length)).to(device)
    # split_type: 区分句子 A (0) 和 句子 B (1)
    split_type = torch.zeros(bs, seq_length, dtype=torch.long).to(device=device)
    split_type[:, 25:50] = 1

    official_Bert = OfficialBertModel.from_pretrained(r'F:\八斗学院\第六周 语言模型\bert-base-chinese').to(device)
    print("--- 开始权重迁移 ---")
    official_Bert_state_dict = official_Bert.state_dict()
    my_model_state_dict = model.state_dict()
    my_model_state_dict['embeddings.word_embeddings.weight'].copy_(official_Bert_state_dict['embeddings.word_embeddings.weight'])
    my_model_state_dict['embeddings.split_embeddings.weight'].copy_(official_Bert_state_dict['embeddings.token_type_embeddings.weight'])
    my_model_state_dict['embeddings.position_embeddings.weight'].copy_(official_Bert_state_dict['embeddings.position_embeddings.weight'])
    my_model_state_dict['embeddings.layernorm.weight'].copy_(official_Bert_state_dict['embeddings.LayerNorm.weight'])
    my_model_state_dict['embeddings.layernorm.bias'].copy_(official_Bert_state_dict['embeddings.LayerNorm.bias'])
    
    
    for i in range(nums_encoder_layer):
        my_model_state_dict[f'encoder_layers.{i}.self_atten.q_linear.weight'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.attention.self.query.weight'])
        my_model_state_dict[f'encoder_layers.{i}.self_atten.q_linear.bias'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.attention.self.query.bias'])
        my_model_state_dict[f'encoder_layers.{i}.self_atten.k_linear.weight'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.attention.self.key.weight'])
        my_model_state_dict[f'encoder_layers.{i}.self_atten.k_linear.bias'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.attention.self.key.bias'])
        my_model_state_dict[f'encoder_layers.{i}.self_atten.v_linear.weight'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.attention.self.value.weight'])
        my_model_state_dict[f'encoder_layers.{i}.self_atten.v_linear.bias'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.attention.self.value.bias'])
        my_model_state_dict[f'encoder_layers.{i}.self_atten.cat_output_linear.weight'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.attention.output.dense.weight'])
        my_model_state_dict[f'encoder_layers.{i}.self_atten.cat_output_linear.bias'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.attention.output.dense.bias'])
        # laynorm1
        my_model_state_dict[f'encoder_layers.{i}.norm1.weight'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.weight'])
        my_model_state_dict[f'encoder_layers.{i}.norm1.bias'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.bias'])
        # ffn
        my_model_state_dict[f'encoder_layers.{i}.ffn.linear1.weight'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.intermediate.dense.weight'])
        my_model_state_dict[f'encoder_layers.{i}.ffn.linear1.bias'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.intermediate.dense.bias'])
        my_model_state_dict[f'encoder_layers.{i}.ffn.linear2.weight'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.output.dense.weight'])
        my_model_state_dict[f'encoder_layers.{i}.ffn.linear2.bias'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.output.dense.bias'])
        # laynorm2
        my_model_state_dict[f'encoder_layers.{i}.norm2.weight'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.output.LayerNorm.weight'])
        my_model_state_dict[f'encoder_layers.{i}.norm2.bias'].copy_(official_Bert_state_dict[f'encoder.layer.{i}.output.LayerNorm.bias'])
    
	# pooler
    my_model_state_dict[f'pooler.0.weight'].copy_(official_Bert_state_dict[f'pooler.dense.weight'])
    my_model_state_dict[f'pooler.0.bias'].copy_(official_Bert_state_dict[f'pooler.dense.bias'])
    # 更新参数到模型中
    model.load_state_dict(my_model_state_dict)
    print("--- 权重迁移完成 ---")
    
    
    model.eval()
    official_Bert.eval()
    print("----------------------")
    with torch.no_grad():
        # 你的模型输出
        my_last_hidden, my_pooler = model(input_ids, split_type)
        # 官方模型输出
        official_last_hidden, official_pooler = official_Bert(input_ids, split_type)
    print(f"我的模型 last_hidden 形状: {my_last_hidden.shape}")
    print(f"官方模型 last_hidden 形状: {official_last_hidden.shape}")
    print(f"我的模型 pooler 形状: {my_pooler.shape}")
    print(f"官方模型 pooler 形状: {official_pooler.shape}")
    
    print(f"我的模型 last_hidden: {my_last_hidden}")
    print(f"官方模型 last_hidden: {official_last_hidden}")
    print(f"我的模型 pooler: {my_pooler}")
    print(f"官方模型 pooler: {official_pooler}")
    
    
    
    
    
    
