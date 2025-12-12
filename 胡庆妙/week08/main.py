# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序：使用三元组损失完成文本匹配模型训练
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据(必须先加载训练数据，再加载模型，因为加载模型时依赖vocab_size)
    train_data = load_data(config["train_data_path"], config)

    # 加载模型
    model = SiameseNetwork(config)
    cuda_flag = torch.cuda.is_available()  # 是否使用gpu
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载测试类
    evaluator = Evaluator(config, model, train_data, logger)

    # 开始训练
    for epoch in range(config["num_epochs"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            batch_ques_vec1, batch_ques_vec2, batch_ques_vec3 = batch_data

            optimizer.zero_grad()  # 梯度归零
            loss = model(batch_ques_vec1, batch_ques_vec2, batch_ques_vec3)  # 计算loss
            train_loss.append(loss.item())

            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

        logger.info("epoch average loss: %.4f" % np.mean(train_loss))
        evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    main(Config)
