import torch
import torch.nn as nn
import math
import numpy as np
from transformers import BertModel


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout_prob=0.1):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} is not a multiple of num heads {num_attention_heads}")

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob=0.1):
        super(BertLayer, self).__init__()
        self.attention = BertSelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.attention_dropout = nn.Dropout(dropout_prob)

        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act = nn.GELU()

        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        self.output_dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + hidden_states)

        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.intermediate_act(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)

        return layer_output

class BertPooler(nn.Module):
    def __init__(self, hidden_size=768):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModelComplete(nn.Module):
    def __init__(self, vocab_size=21128, hidden_size=768, num_hidden_layers=6,
                 num_attention_heads=12, intermediate_size=3072,
                 max_position_embeddings=512, type_vocab_size=2, dropout_prob=0.1):
        super(BertModelComplete, self).__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings,
                                       type_vocab_size, dropout_prob)

        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
            for _ in range(num_hidden_layers)
        ])

        self.pooler = BertPooler(hidden_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        hidden_states = embedding_output
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, extended_attention_mask)

        pooled_output = self.pooler(hidden_states)

        return hidden_states, pooled_output

def load_weights_from_state_dict(model, state_dict):
    model.embeddings.word_embeddings.weight.data = state_dict["embeddings.word_embeddings.weight"].clone()
    model.embeddings.position_embeddings.weight.data = state_dict["embeddings.position_embeddings.weight"].clone()
    model.embeddings.token_type_embeddings.weight.data = state_dict["embeddings.token_type_embeddings.weight"].clone()
    model.embeddings.LayerNorm.weight.data = state_dict["embeddings.LayerNorm.weight"].clone()
    model.embeddings.LayerNorm.bias.data = state_dict["embeddings.LayerNorm.bias"].clone()

    for i, layer in enumerate(model.layers):
        layer.attention.query.weight.data = state_dict[f"encoder.layer.{i}.attention.self.query.weight"].clone()
        layer.attention.query.bias.data = state_dict[f"encoder.layer.{i}.attention.self.query.bias"].clone()
        layer.attention.key.weight.data = state_dict[f"encoder.layer.{i}.attention.self.key.weight"].clone()
        layer.attention.key.bias.data = state_dict[f"encoder.layer.{i}.attention.self.key.bias"].clone()
        layer.attention.value.weight.data = state_dict[f"encoder.layer.{i}.attention.self.value.weight"].clone()
        layer.attention.value.bias.data = state_dict[f"encoder.layer.{i}.attention.self.value.bias"].clone()

        layer.attention_output.weight.data = state_dict[f"encoder.layer.{i}.attention.output.dense.weight"].clone()
        layer.attention_output.bias.data = state_dict[f"encoder.layer.{i}.attention.output.dense.bias"].clone()
        layer.attention_layer_norm.weight.data = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"].clone()
        layer.attention_layer_norm.bias.data = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].clone()

        layer.intermediate.weight.data = state_dict[f"encoder.layer.{i}.intermediate.dense.weight"].clone()
        layer.intermediate.bias.data = state_dict[f"encoder.layer.{i}.intermediate.dense.bias"].clone()

        layer.output.weight.data = state_dict[f"encoder.layer.{i}.output.dense.weight"].clone()
        layer.output.bias.data = state_dict[f"encoder.layer.{i}.output.dense.bias"].clone()
        layer.output_layer_norm.weight.data = state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"].clone()
        layer.output_layer_norm.bias.data = state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"].clone()

    model.pooler.dense.weight.data = state_dict["pooler.dense.weight"].clone()
    model.pooler.dense.bias.data = state_dict["pooler.dense.bias"].clone()

def main():
    bert = BertModel.from_pretrained(r"E:\BaiduNetdiskDownload\第六周 语言模型\bert-base-chinese", return_dict=False)
    state_dict = bert.state_dict()
    bert.eval()

    x = np.array([2450, 15486, 102, 2110])
    torch_x = torch.LongTensor([x])

    with torch.no_grad():
        torch_sequence_output, torch_pooler_output = bert(torch_x)

    model = BertModelComplete(num_hidden_layers=12)
    load_weights_from_state_dict(model, state_dict)
    model.eval()

    with torch.no_grad():
        diy_sequence_output, diy_pooler_output = model(torch_x)

    print(diy_sequence_output)
    print(torch_sequence_output)

if __name__ == "__main__":
    main()