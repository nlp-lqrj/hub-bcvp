import torch
import torch.nn.functional as F
from torch import nn


class Multiheadattention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_size//num_heads

        self.qlinear = nn.Linear(hidden_size, hidden_size)
        self.klinear = nn.Linear(hidden_size, hidden_size)
        self.vlinear = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size,eps=1e-12)



    def forward(self, x, mask= None):
        B,N,C = x.shape

        q = self.qlinear(x).view(B,self.num_heads,N,self.hidden_dim)
        k = self.klinear(x).view(B,self.num_heads,N,self.hidden_dim)
        v = self.vlinear(x).view(B,self.num_heads,N,self.hidden_dim)

        att = q @ k.transpose(-2,-1)/((self.num_heads) ** (-0.5))

        # print(att.shape,mask.shape)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            att = att.masked_fill(mask == 0, -1e9)
        # att = self.layer_norm(att)
        att = F.softmax(att,dim=-1)




        x = att @  v   ## ( B ,heads, N, N) @ (B,heads,N,hidden_size)

        x = x.transpose(1,2).reshape(B, N,C)

        return x

class Bertlayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.Mattention = Multiheadattention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self,x, mask= None):
         x = x + self.Mattention(self.norm1(x), mask)
         x = x + self.mlp(self.norm1(x))

         return x



class BertEmbedding(nn.Module):

    def __init__(self, vocab_size,max_len,hidden_size):
        super().__init__()

        self.tokens = nn.Embedding(vocab_size,hidden_size)
        self.position = nn.Embedding(max_len,hidden_size)
        self.segment = nn.Embedding(2,hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, position_ids=None, segment_ids=None):
        # batch_size, seq_len  =  input_ids.size()



        x_token = self.tokens(input_ids)
        x_position = self.position(position_ids)
        x_segment = self.segment(segment_ids)

        x_embedding = x_token + x_segment + x_position
        x_embedding = self.layer_norm(x_embedding)
        return x_embedding




class bertModel(nn.Module):
    def __init__(self, vocab_size,max_len,hidden_size,num_heads):
        super().__init__()

        self.layers = 12

        self.embeddings = BertEmbedding(vocab_size,max_len,hidden_size)

        self.layers = nn.ModuleList(
            Bertlayer(hidden_size,num_heads) for _ in range(self.layers)
        )


    def forward(self,input_ids, position_ids,segment_ids, attention_mask):
        x = self.embeddings(input_ids, position_ids,segment_ids)
        for layer in self.layers:
            x = layer(x, mask=attention_mask)

        return x

vocab_size = 30522  # BERT默认词汇表大小
hidden_size = 768
num_heads = 12
num_layers = 12
max_seq_len = 512


batch_size = 16
seq_len = 90


input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)  # 随机token索引
segment_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)  # 全属于句子A
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
attention_mask = torch.ones((batch_size, seq_len), dtype=torch.float)  # 无padding，全有效


bertModel =  bertModel(vocab_size,max_seq_len,hidden_size,num_heads)

print(input_ids.shape)
output = bertModel(input_ids, position_ids,segment_ids, attention_mask)


print(output.shape)





