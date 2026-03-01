# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import torch.nn as nn
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

# 设置随机种子
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel

    #大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
    
    model = get_peft_model(model, peft_config)

    if tuning_tactics == "lora_tuning":
        # lora配置会冻结原始模型中的所有层的权重，不允许其反传梯度
        # 但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置
        for param in model.get_submodule("model").get_submodule("classifier").parameters():
            param.requires_grad = True

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            # 确保标签值在有效范围内
            # 创建一个掩码，只保留有效的标签值
            mask = (labels != -1)
            
            # 使用新的模型调用方式，先获取logits
            outputs = model(input_id)
            # 处理不同的返回格式
            if isinstance(outputs, tuple):
                # 对于返回元组的情况，通常第一个元素是logits
                logits = outputs[0]
            else:
                # 对于返回对象的情况，直接获取logits属性
                logits = outputs.logits
            
            # 只计算有效的标签部分的损失
            if mask.sum() > 0:
                # 获取有效的logits和标签
                valid_logits = logits[mask]
                valid_labels = labels[mask]
                # 确保标签值在有效范围内
                valid_labels = torch.clamp(valid_labels, 0, config["class_num"] - 1)
                # 计算损失
                loss = nn.CrossEntropyLoss()(valid_logits, valid_labels)
            else:
                # 如果没有有效的标签，损失为0
                loss = torch.tensor(0.0, requires_grad=True)
                if cuda_flag:
                    loss = loss.cuda()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "%s.pth" % tuning_tactics)
    save_tunable_parameters(model, model_path)  #保存模型权重
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)
