# -*- coding: utf-8 -*-
"""使用 HuggingFace Trainer 进行 LoRA + bert-base-chinese（12 层）NER 训练。

- 复用现有的 ner_data/ 数据与 schema.json
- 指标口径复刻旧版 evaluate.py 的正则解码（regex decode）
- 不使用 CRF（仅使用 token classification 头）
- 支持仅 CPU & GPU

"""

import argparse
import json
import os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model

from hf_dataset_char_bert import CharBertNerDataset
from hf_metrics_regex import compute_metrics, set_eval_sentences


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="bert-base-chinese")
    p.add_argument("--train_path", type=str, default=os.path.join("ner_data", "train"))
    p.add_argument("--dev_path", type=str, default=os.path.join("ner_data", "dev"))
    p.add_argument("--schema_path", type=str, default=os.path.join("ner_data", "schema.json"))
    p.add_argument("--output_dir", type=str, default=os.path.join("outputs", "lora_bert_ner"))
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)

    # 低秩适配超参数
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="query,key,value",
                   help="用逗号分隔的模块名子串，例如 bert-base 可用 'query,key,value'。")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.schema_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_cfg)

    train_ds = CharBertNerDataset(
        data_path=args.train_path,
        schema_path=args.schema_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    dev_ds = CharBertNerDataset(
        data_path=args.dev_path,
        schema_path=args.schema_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # 为旧版“正则解码”指标口径提供原句（按样本顺序对齐）
    set_eval_sentences(dev_ds.sentences)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    fp16 = bool(torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=fp16,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 若启用“训练结束自动加载最优模型”，此时已加载最优的适配器权重
    trainer.save_model(args.output_dir)

    # 同时保存分词器，便于复现与复用
    tokenizer.save_pretrained(args.output_dir)

    # 额外写入说明文件，记录保存内容
    with open(os.path.join(args.output_dir, "README_SAVED.txt"), "w", encoding="utf-8") as f:
        f.write("Saved via Trainer.save_model(). With PEFT LoRA this directory contains adapter weights/config.\n")
        f.write("Tokenizer saved via tokenizer.save_pretrained().\n")


if __name__ == "__main__":
    main()
