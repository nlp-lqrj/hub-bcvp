# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import Seq2SeqModel, choose_optimizer
from evaluate import Evaluator
from loader import build_dataset, load_vocab

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
Seq2Seq模型主程序
实现基于SFT的title到content的生成任务
"""


def main(save_weight=True):

    #加载模型
    model = Seq2SeqModel(Config)

    # 标识是否使用cuda
    use_cuda = torch.cuda.is_available()
    print(f"是否使用cuda：{use_cuda}")
    if use_cuda:
        model = model.cuda()

    # 获取配置参数, 初始化训练前的数据
    epoch_num = Config["epoch_num"]  # 训练轮数
    batch_size = Config["batch_size"]  # 每次训练样本个数
    train_sample = Config["train_sample"]  # 每轮训练总共训练的样本总数
    max_input_len = Config["max_input_len"]  # 输入序列最大长度(title)
    max_output_len = Config["max_output_len"]  # 输出序列最大长度(content)
    # window_size = Config["window_size"]  # 样本长度
    text_length = Config["text_length"]  # 生成文本长度

    # 加载优化器
    optim = choose_optimizer(Config, model)

    # 加载评估器
    evaluator = Evaluator(Config, model, text_length, logger)


    print("文本词表模型加载完毕，开始训练")
    import sys
    sys.stdout.flush()
    
    # 创建模型保存目录
    if save_weight and not os.path.exists("model"):
        os.makedirs("model")
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_idx in range(int(train_sample / batch_size)):
            # 构建一组训练样本，使用title作为input，content作为output
            x, y = build_dataset(batch_size, model.vocab, max_input_len, max_output_len, model.data_list)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
            if batch_idx % 255 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                sys.stdout.flush()
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        sys.stdout.flush()
        
        # 使用几个示例title进行测试生成
        test_titles = ["阿根廷歹徒抢服装尺码不对拿回店里换", 
                      "国际通用航空大会沈阳飞行家表演队一飞机发生坠机，伤亡不明",
                      "邓亚萍：互联网要有社会担当"]
        for title in test_titles[:2]:  # 每次测试前两个标题
            generated_content = evaluator.generate_content(title, model, model.vocab, max_input_len, max_output_len)
            print(f"\n输入标题: {title}")
            print(f"生成内容: {generated_content}")
            sys.stdout.flush()
    
    if not save_weight:
        return
    else:
        base_name = os.path.basename(Config["corpus_path"]).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        print(f"\n模型已保存至: {model_path}")

if __name__ == "__main__":
    main(False)