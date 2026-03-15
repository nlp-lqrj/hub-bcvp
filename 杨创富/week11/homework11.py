"""
Seq2Seq模型SFT（Supervised Fine-Tuning）微调示例
使用transformers框架进行监督式微调，适用于T5等Seq2Seq模型
"""

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import numpy as np

# 配置
MODEL_NAME = "./models/t5-small"
OUTPUT_DIR = "./output"
MAX_LENGTH = 512

def create_seq2seq_dataset(num_samples=1000):
    """构造Seq2Seq格式的数据集（例如：中文→英文翻译）"""
    # 基础数据
    base_pairs = [
        {"zh": "我爱自然语言处理", "en": "I love natural language processing"},
        {"zh": "深度学习是机器学习的一个分支", "en": "Deep learning is a branch of machine learning"},
        {"zh": "人工智能正在改变世界", "en": "Artificial intelligence is changing the world"},
        {"zh": "神经网络由神经元组成", "en": "Neural networks consist of neurons"},
        {"zh": "这个模型表现非常好", "en": "This model performs very well"},
        {"zh": "我们需要更多的训练数据", "en": "We need more training data"},
        {"zh": "准确率达到了百分之九十五", "en": "The accuracy reached ninety-five percent"},
        {"zh": "请解释一下这个概念", "en": "Please explain this concept"},
        {"zh": "如何提高模型性能", "en": "How to improve model performance"},
        {"zh": "这是一个重要的发现", "en": "This is an important discovery"}
    ]
    
    # 扩展数据
    data = []
    for i in range(num_samples):
        pair = base_pairs[i % len(base_pairs)]
        # 添加一些变化
        variations = [
            (f"翻译：{pair['zh']}", f"Translate: {pair['en']}"),
            (f"请翻译：{pair['zh']}", f"Please translate: {pair['en']}"),
            (f"将以下中文翻译成英文：{pair['zh']}", f"Translate Chinese to English: {pair['en']}"),
            (pair['zh'], pair['en']),  # 直接对
            (f"{pair['zh']}（翻译）", pair['en'])
        ]
        
        input_text, target_text = variations[i % len(variations)]
        data.append({
            "input_text": input_text,
            "target_text": target_text
        })
    
    return data

def preprocess_function(examples, tokenizer):
    """预处理函数，将文本转换为模型输入"""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    # 设置标签
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    # 解码预测结果
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # 解码标签（将-100替换为pad_token_id）
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 这里可以计算BLEU、ROUGE等指标
    # 简化为计算精确匹配
    exact_match = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l) / len(decoded_preds)
    
    return {"exact_match": exact_match}

def load_model_and_tokenizer():
    """加载模型和分词器"""
    print(f"正在加载模型: {MODEL_NAME}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    
    # 调整模型大小以节省内存（如果需要）
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()
    
    return model, tokenizer

def main():
    """主训练函数"""
    print("=" * 50)
    print("开始Seq2Seq模型微调训练")
    print("=" * 50)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 创建数据集
    print("\n正在构造训练数据...")
    raw_data = create_seq2seq_dataset(num_samples=200)
    dataset = Dataset.from_list(raw_data)
    
    # 划分训练集和验证集
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"训练样本数量: {len(train_dataset)}")
    print(f"验证样本数量: {len(eval_dataset)}")
    
    # 预处理数据集
    print("\n正在预处理数据...")
    tokenized_train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    

    training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    eval_steps=50,
    save_steps=100,
    logging_steps=10,
    predict_with_generate=True,
    load_best_model_at_end=False,  # 设置为 False
    report_to="none",
    )
    
    # 创建训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("\n开始训练...")
    train_result = trainer.train()
    
    # 保存模型
    print(f"\n训练完成，正在保存模型到 {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    print("=" * 50)
    print("训练完成！")
    print("=" * 50)
    
    # 测试模型
    print("\n测试模型输出：")
    test_inputs = [
        "翻译：我爱自然语言处理",
        "深度学习是机器学习的一个分支",
        "如何提高模型性能"
    ]
    
    for input_text in test_inputs:
        inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"输入: {input_text}")
        print(f"输出: {output_text}")
        print("-" * 30)

if __name__ == "__main__":
    main()
