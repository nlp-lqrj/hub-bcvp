import torch
from transformers import BertModel


def calculate_bert_parameters(model_name='bert-base-uncased'):
    """
    计算BERT模型的参数量

    Args:
        model_name: BERT模型名称，默认为bert-base-uncased
    """
    # 加载预训练的BERT模型
    model = BertModel.from_pretrained(model_name)

    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())

    # 计算可训练参数量（通常所有参数都是可训练的）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"BERT模型: {model_name}")
    print("=" * 50)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"参数量大小: {total_params / 1e6:.1f}M")

    # 按模块分解参数量
    print("\n各模块参数量分解:")
    print("-" * 30)

    module_params = {}
    for name, module in model.named_children():
        module_param = sum(p.numel() for p in module.parameters())
        module_params[name] = module_param
        print(f"{name:15}: {module_param:>12,} ({module_param / total_params * 100:.1f}%)")

    return total_params, trainable_params, module_params


def detailed_bert_analysis(model_name='bert-base-uncased'):
    """
    更详细的BERT参数量分析
    """
    model = BertModel.from_pretrained(model_name)

    print("BERT详细参数量分析")
    print("=" * 60)

    # 嵌入层分析
    embeddings = model.embeddings
    emb_params = sum(p.numel() for p in embeddings.parameters())
    print(f"\n1. 嵌入层:")
    print(
        f"   - 词嵌入: {embeddings.word_embeddings.num_embeddings} × {embeddings.word_embeddins.embedding_dim} = {embeddings.word_embeddings.weight.numel():,}")
    print(
        f"   - 位置嵌入: {embeddings.position_embeddings.num_embeddings} × {embeddings.position_embeddings.embedding_dim} = {embeddings.position_embeddings.weight.numel():,}")
    print(
        f"   - 段落嵌入: {embeddings.token_type_embeddings.num_embeddings} × {embeddings.token_type_embeddings.embedding_dim} = {embeddings.token_type_embeddings.weight.numel():,}")
    print(f"   - LayerNorm: {sum(p.numel() for p in embeddings.LayerNorm.parameters()):,}")
    print(f"   嵌入层总计: {emb_params:,}")

    # 编码器层分析
    encoder = model.encoder
    print(f"\n2. 编码器 (共{len(encoder.layer)}层):")

    # 分析单层的参数量
    if len(encoder.layer) > 0:
        single_layer = encoder.layer[0]
        layer_params = sum(p.numel() for p in single_layer.parameters())

        # 自注意力机制
        attention = single_layer.attention.self
        attention_params = sum(p.numel() for p in attention.parameters())

        # 输出层
        output = single_layer.attention.output
        output_params = sum(p.numel() for p in output.parameters())

        # 前馈网络
        intermediate = single_layer.intermediate
        intermediate_params = sum(p.numel() for p in intermediate.parameters())

        # 输出层
        output_layer = single_layer.output
        output_layer_params = sum(p.numel() for p in output_layer.parameters())

        print(f"   单层参数量: {layer_params:,}")
        print(f"   - 自注意力: {attention_params:,}")
        print(f"   - 注意力输出: {output_params:,}")
        print(f"   - 前馈网络: {intermediate_params:,}")
        print(f"   - 前馈输出: {output_layer_params:,}")

        total_encoder_params = layer_params * len(encoder.layer)
        print(f"   编码器总计: {total_encoder_params:,}")


def compare_bert_variants():
    """
    比较不同BERT变体的参数量
    """
    variants = [
        'bert-base-uncased',
        'bert-large-uncased',
        'bert-base-chinese',
        'bert-base-multilingual-cased'
    ]

    print("BERT不同变体参数量比较")
    print("=" * 60)

    results = {}
    for variant in variants:
        try:
            model = BertModel.from_pretrained(variant)
            total_params = sum(p.numel() for p in model.parameters())
            results[variant] = total_params
            print(f"{variant:25}: {total_params:>12,} ({total_params / 1e6:6.1f}M)")
        except Exception as e:
            print(f"{variant:25}: 加载失败 - {e}")

    return results


if __name__ == "__main__":
    # 基本参数量计算
    print("BERT参数量计算")
    print("=" * 50)

    total_params, trainable_params, module_params = calculate_bert_parameters()
