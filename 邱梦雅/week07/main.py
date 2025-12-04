# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import pandas as pd

#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.setLevel(logging.ERROR)

"""
模型训练主程序
"""


seed = Config["seed"] 
random.seed(seed)   #设置Python内置random模块的随机种子，影响random.random()、random.randint()等函数的随机数生成
np.random.seed(seed)
torch.manual_seed(seed)  # 设置PyTorch的随机种子，影响PyTorch CPU上的随机数生成
torch.cuda.manual_seed_all(seed)   # 设置所有GPU的随机种子，如果使用多GPU训练，这确保所有GPU上的随机数生成都是可重复的
torch.cuda.empty_cache()
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# torch.cuda.set_per_process_memory_fraction(0.9, device=0)  # 设置最大使用的显存比例
# torch.cuda.set_memory_alloc_conf(max_split_size_mb=64)  # 设置最大分配单元


def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
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
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()            #梯度归零
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)  #计算loss
            loss.backward()                  #计算梯度
            optimizer.step()                 #更新权重

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    results = list()
    # Group 1：对比非预训练模型
    model_list_1 = ["fast_text", "cnn", "gated_cnn", "rnn", "lstm", "gru", "rcnn", "stack_gated_cnn"]
    # Group 2：对比预训练模型
    model_list_2 = ["bert", "bert_lstm", "bert_cnn", "bert_mid_layer"]
    # Group 3：对比非预训练模型和预训练模型
    model_list_3 = ["gated_cnn", "lstm", "rcnn", "bert", "bert_mid_layer"]

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    for model in model_list_3:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4, 1e-5]:
            Config["learning_rate"] = lr
            for hidden_size in [128, 256]:
                Config["hidden_size"] = hidden_size
            # for opt in ["adam", "sgd"]:
            #     Config["optimizer"] = opt
                for max_length in [30, 50]:
                    Config["max_length"] = max_length
                    for batch_size in [64, 128]:
                        Config["batch_size"] = batch_size
                        for pooling_style in ["avg", 'max']:
                            Config["pooling_style"] = pooling_style
                            acc = main(Config)
                            res = {
                                "model_type": model,
                                "max_length": max_length,
                                "hidden_size": Config["hidden_size"] if model == 'bert' else hidden_size,
                                # "kernel_size": kernel_size,
                                "pooling_style": pooling_style,
                                "batch_size": batch_size,
                                "learning_rate": lr,
                                "optimizer": Config["optimizer"],
                                "epoch": Config["epoch"],
                                "vocab_size": Config["vocab_size"],
                                # "num_layers": Config["num_layers"],
                                "accuracy": "{:.6f}".format(acc)
                            }
                            results.append(res)
                            print("最后一轮准确率：", acc, "当前配置：", Config)


    # 保存结果
    # 一次性创建DataFrame
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    table_results = pd.DataFrame(results)

    # 导出为Markdown
    save_file = f'text_classification_result_group_3.md'
    table_results.to_markdown(save_file, index=False, tablefmt='github')
    print(f"结果已保存到 {save_file}")

