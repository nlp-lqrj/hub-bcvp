# -*- coding: utf-8 -*-
import time

import torch
import os
import numpy as np
import logging

from torch import nn

from config import Config
from model import setup_rola_model, reset_requires_grad, save_tunable_params, choose_optimizer
from evaluate import Evaluator
import loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序: 基于bert做LoRA微调，实现ner的序列标注任务。
"""


def main(config):
    # 保存模型参数的目录
    if not os.path.isdir(config["model_param_path"]):
        os.mkdir(config["model_param_path"])

    schema = loader.load_schema(Config["schema_path"])

    # 支持RoLA的模型
    model = setup_rola_model(config["pretrain_model_path"], len(schema))
    # 设定可微调的参数
    reset_requires_grad(model)

    cuda_flag = torch.cuda.is_available()  # 是否使用gpu
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载训练数据
    train_data = loader.load_data(config["train_data_path"], config)

    # 加载验证类
    evaluator = Evaluator(config, model, logger)

    begin_time = time.time()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 开始训练...")
    epoch = 0
    for epoch in range(config["epoch_num"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []

        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            # input_ids: [batch_size, seq_len, num_labels], label_ids: [batch_size, seq_len]
            input_ids, label_ids = batch_data
            attention_mask = (input_ids != 0).long()

            logits = model(input_ids, attention_mask=attention_mask)[0]  # [batch_size, seq_len, num_labels]
            loss = nn.CrossEntropyLoss(ignore_index=-1)(logits.view(-1, logits.shape[-1]), label_ids.view(-1))
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 梯度归零

            train_loss.append(loss.item())
            if (index + 1) % int(len(train_data) / 2) == 0:
                logger.info("batch loss %.4f" % loss.item())

        logger.info("epoch average loss: %.4f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)

    # 保存模型微调部分的参数
    model_param_path = os.path.join(config["model_param_path"], "ner_LoRA.pth")
    save_tunable_params(model, model_param_path)

    spent_time = round((time.time() - begin_time), 2)  # 训练及验证耗时
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 训练结束, 耗时: {spent_time}秒")


if __name__ == "__main__":
    main(Config)
