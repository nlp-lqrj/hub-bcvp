# 张智睿-提交第13周作业： NER 项目（新增 LoRA + BERT Trainer 训练链路）

原有的 BiLSTM(+CRF) 代码与数据格式保持不变；新增了一条 **LoRA + bert-base-chinese(12层) + Trainer** 的训练链路。

## 训练（CPU 或单卡均可）
在项目根目录执行：

```bash
train_lora_bert_trainer.py \
  --model_name bert-base-chinese \
  --max_length 128 \
  --epochs 10 \
  --train_batch_size 16 \
  --eval_batch_size 16
```

训练输出默认在：`outputs/lora_bert_ner/`
