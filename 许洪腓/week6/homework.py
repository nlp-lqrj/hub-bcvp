import torch 
import torch.nn as nn 
import math 

class BertEncoderLayer(nn.Module):
    def __init__(self, input_dim=768, hidden_size=768, num_attention_heads=12, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads =  num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout = dropout

        # 1.多头注意力的QKV层
        self.qkv_proj = nn.Linear(input_dim, 3*hidden_size)         # w: 768*3*768 + 3*768
        self.attention_output_proj = nn.Linear(hidden_size, hidden_size)    # 768*768 + 768
        self.attention_dropout = nn.Dropout(dropout)

        # 2.ffn的前馈网络层
        self.ffn_layer1 = nn.Linear(hidden_size, 4*hidden_size)     # w: 768 * 4*768 + 4*768
        self.activation = nn.functional.gelu    
        self.ffn_layer2 = nn.Linear(4*hidden_size, hidden_size)     # w: 4*768 * 768 + 768
        self.ffn_out_dropout = nn.Dropout(dropout)
        
        # 3.层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-12)     # w: 768*2
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-12)     # w: 768*2

        # 4.残差连接时变换项的dropout
        self.residual_dropout = nn.Dropout(dropout)

    def multi_heads_attention(self, x):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1,2)

        attention_scores = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        context_vec = torch.matmul(attention_weights, v)

        # 拼接多头
        context_vec = context_vec.transpose(1,2).contiguous()
        context_vec = context_vec.view(batch_size, seq_len, self.hidden_size)
        attention_output = self.attention_output_proj(context_vec)
        attention_output = self.residual_dropout(attention_output)

        return attention_output
        
    def forward(self, x):
        attention_output = self.multi_heads_attention(x)

        x1 = x + attention_output 
        x1 = self.layer_norm1(x1)

        ffn_output = self.ffn_layer1(x1)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_out_dropout(ffn_output)
        ffn_output = self.ffn_layer2(ffn_output)

        x2 = x1 + ffn_output
        output = self.layer_norm2(x2)

        return output 
    
if __name__ == '__main__':
    batch_size, seq_len, input_dim = 2, 10, 768
    x = torch.randn(batch_size, seq_len, input_dim)
    bert_encoder_layer = BertEncoderLayer()
    y = bert_encoder_layer(x)
    print(f"输入的Shape{x.shape}")
    print(f"输出的Shape{y.shape}")
    params_num = (768*3*768 + 3*768) + (768*768 + 768) + (768 * 4*768 + 4*768) + (4*768 * 768 + 768) + (768*2) + (768*2)
    print(f"成功运行，要训练的参数个数为{params_num}，{params_num/10e8}B")

    
