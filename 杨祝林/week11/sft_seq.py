import torch
import datasets
import transformers
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import warnings

warnings.filterwarnings("ignore")

# ====================== 1. 基础配置 ======================
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 模型和训练参数
MODEL_NAME = "t5-small"  # 轻量级Seq2Seq模型，适合入门
MAX_INPUT_LENGTH = 128  # 输入序列最大长度
MAX_OUTPUT_LENGTH = 64  # 输出序列最大长度
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5


# ====================== 2. 构建示例数据集 ======================
# 模拟SFT训练数据：(指令+输入, 目标输出) 格式
def create_demo_dataset():
    # 典型的Seq2Seq SFT数据格式
    data = {
        "instruction": [
            "将以下句子翻译成英文：",
            "总结以下文本的核心内容：",
            "回答以下问题：",
            "将以下句子翻译成中文：",
            "根据上下文续写："
        ],
        "input": [
            "今天天气很好，适合出门散步",
            "人工智能技术正在快速发展，深度学习是其中的核心分支，已广泛应用于图像识别、自然语言处理等领域",
            "什么是Seq2Seq模型？",
            "The quick brown fox jumps over the lazy dog",
            "从前有座山，山里有座庙"
        ],
        "output": [
            "The weather is nice today, perfect for a walk outside",
            "人工智能技术发展迅速，深度学习作为核心分支，应用于图像识别和自然语言处理等领域",
            "Seq2Seq模型是一种编码器-解码器架构的深度学习模型，用于将一个序列转换为另一个序列，广泛应用于机器翻译、对话生成等任务",
            "敏捷的棕色狐狸跳过了懒狗",
            "从前有座山，山里有座庙，庙里有个老和尚在讲故事"
        ]
    }

    # 组合指令和输入作为模型输入
    dataset = datasets.Dataset.from_dict(data)
    dataset = dataset.map(lambda x: {"text": f"{x['instruction']} {x['input']}"})
    return dataset.train_test_split(test_size=0.2)  # 划分训练/测试集


# ====================== 3. 数据预处理 ======================
def preprocess_function(examples, tokenizer):
    """
    对数据进行tokenization处理，适配Seq2Seq模型的输入格式
    """
    # 处理输入
    inputs = examples["text"]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # 处理标签（输出）
    labels = tokenizer(
        examples["output"],
        max_length=MAX_OUTPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # 将标签填充部分设为-100（PyTorch CrossEntropyLoss会忽略-100）
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    return model_inputs


# ====================== 4. 主训练流程 ======================
def main():
    # 1. 加载数据集
    dataset = create_demo_dataset()
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # 2. 加载tokenizer和模型
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    # 3. 数据预处理
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )

    # 4. 数据collator（处理批次数据）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=MAX_INPUT_LENGTH
    )

    # 5. 训练参数配置
    training_args = TrainingArguments(
        output_dir="./sft_seq2seq_results",  # 输出目录
        num_train_epochs=EPOCHS,  # 训练轮数
        per_device_train_batch_size=BATCH_SIZE,  # 每个设备的批次大小
        per_device_eval_batch_size=BATCH_SIZE,  # 验证批次大小
        learning_rate=LEARNING_RATE,  # 学习率
        logging_dir="./logs",  # 日志目录
        logging_steps=10,  # 日志打印步数
        evaluation_strategy="epoch",  # 每轮验证一次
        save_strategy="epoch",  # 每轮保存一次
        load_best_model_at_end=True,  # 训练结束加载最优模型
        metric_for_best_model="loss",  # 基于loss选择最优模型
        fp16=torch.cuda.is_available(),  # 混合精度训练（GPU可用时开启）
        weight_decay=0.01,  # 权重衰减
        report_to="none"  # 不使用wandb等日志工具
    )

    # 6. 构建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 7. 开始训练
    print("开始SFT训练...")
    trainer.train()

    # 8. 保存模型
    model.save_pretrained("./sft_t5_model")
    tokenizer.save_pretrained("./sft_t5_model")
    print("模型保存完成！")

    # ====================== 5. 推理测试 ======================
    def generate_response(input_text):
        """使用训练后的模型生成回复"""
        # 预处理输入
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length"
        ).to(device)

        # 生成输出
        outputs = model.generate(
            **inputs,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=4,  # 束搜索
            temperature=0.7,  # 随机性
            top_p=0.9,
            repetition_penalty=1.2
        )

        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    # 测试示例
    test_inputs = [
        "将以下句子翻译成英文：我喜欢学习Python编程",
        "总结以下文本的核心内容：Seq2Seq模型由编码器和解码器组成，编码器负责处理输入序列，解码器负责生成输出序列"
    ]

    print("\n===== 推理测试 =====")
    for input_text in test_inputs:
        response = generate_response(input_text)
        print(f"输入: {input_text}")
        print(f"输出: {response}\n")


if __name__ == "__main__":
    main()
