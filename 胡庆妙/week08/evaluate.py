# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, train_data, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.test_data = load_data(config["test_data_path"], config, shuffle=False)
        self.train_data = train_data  # 效果测试需要训练集当做知识库
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    # 将知识库中的问题向量化，为匹配做准备，最终得到shape为[常用问的数量, embed_size] 的张量。
    # 每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    def knwb_to_vector(self):
        self.ques_id_to_std_ques_id = {}  # {常用问id: 标准问id}
        self.ques_idvec_list = []  # 所有常用问的向量

        # {标准问id: [常用问1的向量, 常用问2的向量], ...}
        for std_ques_id, ques_idvecs in self.train_data.dataset.knwb.items():
            for ques_idvec in ques_idvecs:
                # 记录常用问id到标准问题id的映射，常用问id就是ques_idvec_list中的问题索引号
                self.ques_id_to_std_ques_id[len(self.ques_idvec_list)] = std_ques_id
                self.ques_idvec_list.append(ques_idvec)

        with torch.no_grad():
            question_matrixs = torch.stack(self.ques_idvec_list, dim=0)  # shape: [常用问的数量, sentence_len]
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()  # 移动到GPU上运行

            # [常用问的数量, sentence_len] -> [常用问的数量, embed_size]
            self.knwb_vectors = self.model(question_matrixs)

            # 将所有向量都作归一化
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)  # [常用问的数量, embed_size]
        return

    def eval(self, epoch):
        self.logger.info("开始测试 第%d轮 的训练效果：" % epoch)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()  # [常用问的数量, embed_size]

        for index, batch_data in enumerate(self.test_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            batch_ques_idvec, batch_std_ques_id = batch_data  # 常用问的向量, [标准问id]
            with torch.no_grad():
                batch_ques_vector = self.model(batch_ques_idvec)  # [batch_size, sen_len] -> [batch_size, embed_size]
            self.write_stats(batch_ques_vector, batch_std_ques_id)

        self.show_stats()
        return

    def write_stats(self, batch_ques_vector, batch_std_ques_id):
        """
        Args:
            batch_ques_vector: [batch_size, embed_size]
            batch_std_ques_id: [batch_size, 1]
        Returns:
        """
        assert len(batch_ques_vector) == len(batch_std_ques_id)
        for ques_vector, label_id in zip(batch_ques_vector, batch_std_ques_id):
            # 通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            # ques_vector: [embed_size] -> [1, embed_size]
            # knwb_vectors: [常用问的数量, embed_size] -> [embed_size, 常用问的数量]
            # [1, embed_size] mm [embed_size, 常用问的数量] -> [1, 常用问的数量]
            res = torch.matmul(ques_vector.unsqueeze(0), self.knwb_vectors.T)

            hit_ques_id = int(torch.argmax(res.squeeze()))  # 取得相似度最大的那个问题向量的编号
            std_ques_id = self.ques_id_to_std_ques_id[hit_ques_id]  # 转换成标准问的编号
            if int(std_ques_id) == int(label_id):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%.4f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return
