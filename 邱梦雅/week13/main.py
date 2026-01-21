# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig
import time

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
log_path = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".log"
handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
logger.addHandler(handler)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)

    # 大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=32,
            lora_alpha=32,    # 缩放因子
            lora_dropout=0.1,
            # target_modules=["query", "key", "value"]   # 在 q、k、v 线性层添加LoRA模块 bert模型中每一个网络层是有名字的 state_dict
            target_modules=["query", "key", "value", "attention.output.dense"]   # 在 q、k、v、o 线性层添加LoRA模块
        )
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

    model = get_peft_model(model, peft_config)  # 配置了LoRA模块的模型
    # print(model.state_dict().keys())   # 在 q、k、v 线性层添加LoRA模块 bert模型中每一个网络层是有名字的 state_dict
    for key, value in model.state_dict().items():
        print(key, value.shape)

    if tuning_tactics == "lora_tuning":
        # lora配置会冻结原始模型中的所有层的权重，不允许其反传梯度，即param.requires_grad = False
        # 但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置
        for name, param in model.get_submodule("model").get_submodule("classify").named_parameters():
            param.requires_grad = True
            print(f"{name} 权重是否参与训练: {param.requires_grad}")
        # 如果设置了crf层，希望crf层也照常训练
        for name, param in model.get_submodule("model").get_submodule("crf_layer").named_parameters():
            param.requires_grad = True
            print(f"{name} 权重是否参与训练: {param.requires_grad}")

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    logger.info("文本词表模型加载完毕，开始训练")
    logger.info(f"epoch num: {config["epoch"]}")
    logger.info(f"batch_size: {config["batch_size"]}")
    logger.info(f"tuning_tactics: {tuning_tactics}")
    logger.info(f"rank: {peft_config.r}")
    logger.info(f"target_modules: {peft_config.target_modules}")
    logger.info(f"learning rate: {optimizer.param_groups[0]['lr']}")
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
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "%s_epoch_%d.pth" % (tuning_tactics, epoch))
    save_tunable_parameters(model, model_path)  #保存模型权重
    # torch.save(model.state_dict(), model_path)
    return model, train_data

# 只保存可训练的参数
# model.parameters() 返回一个 迭代器（iterator），其中包含了模型中所有的参数（torch.nn.Parameter 对象）。返回内容：只有参数的 值（张量）
# model.named_parameters() 返回一个 迭代器，其中的每一项都是一个 元组（tuple），格式为 (name, parameter)。返回内容：参数的 名字（字符串） 和 值（张量） 的配对。
def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")   # 使用.to("cpu")确保参数被移动到CPU内存中（节省GPU内存）
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)

if __name__ == "__main__":
    model, train_data = main(Config)
