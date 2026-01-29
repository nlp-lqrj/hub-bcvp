# -*- coding:utf-8 -*-

import os
import torch
import numpy as np
from loader import load_data
from config import Config
from evaluate import Evaluator
from model import NerClassificationModel
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig
cuda = torch.cuda.is_available()


def load_peft_tuning(model, tuning_type):
    """
    task_type参数说明：用于向PEFT库指明模型要执行的任务类型。它通常不改变底层的算法逻辑，但有时会影响PEFT模型初始化时的内部行为
    SEQ_CLS             序列分类（如情感分析）
    SEQ_2_SEQ_LM        序列到序列语言模型（如翻译、摘要）
    CAUSAL_LM           因果语言模型（如文本生成）
    TOKEN_CLS           令牌分类（如命名实体识别NER）
    QUESTION_ANS        问答
    FEATURE_EXTRACTION  特征提取
    """
    if tuning_type == "lora":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "attention.output.dense"],
            task_type="TOKEN_CLS"
        )
    elif tuning_type == "prompt":
        peft_config = PromptTuningConfig(
            task_type="TOKEN_CLS",
            num_virtual_tokens=10  # 软提示向量的数量（长度）
        )
    else:
        peft_config = PrefixTuningConfig(
            task_type="TOKEN_CLS",
            num_virtual_tokens=10  # 每层前缀向量的数量
        )
        # 将peft配置嵌入模型中，此时peft会冻结原始模型所有权重
    model = get_peft_model(model, peft_config)
    # 但是我们只需要冻结bert中的权重，而分类线性层和crf中的权重需要在训练中随任务更新
    for param in model.get_submodule("model").get_submodule("classifier").parameters():
        param.requires_grad = True
    for param in model.get_submodule("model").get_submodule("crf").parameters():
        param.requires_grad = True

    return model


def main(config):
    model = NerClassificationModel(config)
    # 加载peft参数，并设置非冻结权重
    model = load_peft_tuning(model, config["tuning_type"])
    if cuda:
        print("将模型迁移至GPU")
        model = model.cuda()
    # 训练数据
    train_data = load_data(config, config["train_data_path"])
    optim = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    model.train()
    for epoch in range(config["num_epochs"]):
        print(f"===开始第{epoch + 1}轮训练===")
        watch_loss = []
        count = 0
        for index, batch_data in enumerate(train_data):
            if cuda:
                batch_data = [b.cuda() for b in batch_data]
            batch_x, batch_y = batch_data
            loss = model(batch_x, tags=batch_y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
            # 每训练10%检查一次
            if index % int(len(train_data) / 10) == 0 and index > 0:
                count += 1
                print(f"训练进度：{count / 10:.0%}")
        print(f"第{epoch + 1}轮训练结束，平均loss为：{np.mean(watch_loss):.6f}")
        # 对每轮的训练结果进行测试
        evaluator = Evaluator(config, model)
        evaluator.predict()

    # 加载模型保存路径
    model_save_path = config["model_save_dir"]
    if os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    model_path = os.path.join(model_save_path, config["tuning_type"] + ".pth")

    # 只保存peft权重
    saved_params = {k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad}
    torch.save(saved_params, model_path)


if __name__ == "__main__":
    main(Config)