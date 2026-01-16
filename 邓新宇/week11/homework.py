"""
大语言模型SFT（Supervised Fine-Tuning）微调示例 - Seq2Seq版本
使用trl框架进行监督式微调，适配Encoder-Decoder架构（如T5, mT5, BART）
"""

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig

# 配置
MODEL_NAME = "./models/mt5-small"  # 可替换为 "t5-base", "facebook/bart-base", 或本地路径
OUTPUT_DIR = "./output_seq2seq"
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256


# 构造中文问答数据（分离输入与输出）
def create_chinese_dataset():
    """构造简单的中文问答数据集，返回 source 和 target"""
    data = [
        {
            "instruction": "请介绍一下人工智能。",
            "output": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、感知和自然语言处理等能力。"
        },
        {
            "instruction": "什么是深度学习？",
            "output": "深度学习是机器学习的一个子领域，使用具有多个层（深度）的神经网络来学习数据的复杂模式和表示。它模仿人脑的神经网络结构。"
        },
        {
            "instruction": "Python中的列表和元组有什么区别？",
            "output": "列表（list）是可变的，可以修改、添加或删除元素，使用方括号[]。元组（tuple）是不可变的，一旦创建就不能修改，使用圆括号()。"
        },
        {
            "instruction": "解释一下什么是监督学习。",
            "output": "监督学习是机器学习的一种方法，使用标记的训练数据来训练模型。模型学习输入和输出之间的映射关系，然后可以对新的未标记数据进行预测。"
        },
        {
            "instruction": "如何提高模型的泛化能力？",
            "output": "提高模型泛化能力的方法包括：1) 增加训练数据量和多样性 2) 使用正则化技术（如Dropout、L2正则化）3) 数据增强 4) 交叉验证 5) 防止过拟合。"
        },
        {
            "instruction": "什么是Transformer模型？",
            "output": "Transformer是一种基于注意力机制的神经网络架构，由编码器和解码器组成。它摒弃了RNN和CNN，完全依赖注意力机制来处理序列数据，成为现代NLP的基础架构。"
        },
        {
            "instruction": "请解释一下梯度下降算法。",
            "output": "梯度下降是一种优化算法，用于最小化损失函数。它通过计算损失函数对参数的梯度，然后沿着梯度反方向更新参数，逐步接近最优解。学习率控制每次更新的步长。"
        },
        {
            "instruction": "什么是迁移学习？",
            "output": "迁移学习是将在一个任务或领域上学到的知识应用到另一个相关任务上的技术。它允许模型利用预训练的知识，从而在目标任务上更快地学习和获得更好的性能。"
        },
        {
            "instruction": "如何处理自然语言处理中的文本分类问题？",
            "output": "文本分类的常见步骤包括：1) 文本预处理（分词、去停用词）2) 特征提取（词袋、TF-IDF、词向量）3) 选择分类算法（朴素贝叶斯、SVM、神经网络）4) 训练和评估模型。"
        },
        {
            "instruction": "请介绍一下大语言模型。",
            "output": "大语言模型（LLM）是拥有数十亿甚至千亿参数的深度学习模型，通过在海量文本数据上预训练获得语言理解能力。它们可以执行各种NLP任务，如文本生成、问答、翻译等。"
        }
    ]

    # 返回 source 和 target 字段（符合Seq2Seq格式）
    formatted_data = []
    for item in data:
        formatted_data.append({
            "source": item["instruction"],
            "target": item["output"]
        })

    return formatted_data


def preprocess_function(examples, tokenizer):
    """对数据进行tokenize，适配Seq2Seq训练"""
    inputs = [ex for ex in examples["source"]]
    targets = [ex for ex in examples["target"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_model_and_tokenizer():
    """加载Seq2Seq模型和分词器"""
    print(f"正在加载Seq2Seq模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    return model, tokenizer


def main():
    print("=" * 50)
    print("开始Seq2Seq SFT微调训练")
    print("=" * 50)

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()

    # 创建原始数据
    raw_data = create_chinese_dataset()
    dataset = Dataset.from_list(raw_data)

    print(f"训练样本数量: {len(dataset)}")

    # 预处理数据
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=["source", "target"]
    )

    # 数据整理器（处理labels的-100填充）
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 训练配置（注意：SFTTrainer 在新版本 trl 中对 Seq2Seq 支持有限）
    # 因此我们改用标准 HuggingFace Trainer + 自定义训练循环，或直接使用 SFTTrainer 并指定 format
    # 但更推荐：**不使用 SFTTrainer，而用标准 Trainer** —— 因为 SFTTrainer 主要为 Causal LM 设计

    # === 改用标准 TrainingArguments + Trainer 更稳妥 ===
    from transformers import Trainer

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to=None,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n开始训练...")
    trainer.train()

    print(f"\n训练完成，保存模型到 {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 50)
    print("Seq2Seq SFT 微调完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()