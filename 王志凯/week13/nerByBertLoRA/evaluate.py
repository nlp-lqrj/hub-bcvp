# -*- coding:utf-8 -*-


import re
import torch
import numpy as np
from collections import defaultdict
from loader import load_data


class Evaluator:
    def __init__(self, config, model):
        self.model = model
        self.test_data = load_data(config, config["test_data_path"])
        # 加载test_data时已经初始化过SentenceLabelLoader对象，直接从该对象中获取vocab_dict，不需要重新加载
        self.vocab_dict = self.test_data.dataset.vocab_dict
        self.use_crf = config["use_crf"]
        self.F1_type = config["F1_type"]   # F1的计算方式，0:微观，1:宏观（默认为0）

    def predict(self):
        print("对该轮训练结果进行预测...")
        # 记录每个实体的总数、预测出的实体数，预测正确的实体数
        self.entity_info = {
            "PERSON": defaultdict(int),
            "LOCATION": defaultdict(int),
            "ORGANIZATION": defaultdict(int),
            "TIME": defaultdict(int)
        }
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in self.test_data:
                batch_y_pred = self.model(batch_x)
                # 统计各个实体
                self.countEntity(batch_y, batch_y_pred)
            # 计算准确率、召回率、F1
            self.calculateEntity()

    def countEntity(self, batch_true_labels, batch_pred_labels):
        assert len(batch_pred_labels) == len(batch_true_labels)
        if not self.use_crf:
            # crf返回的是最优的标签序列，而不使用crf时，返回的是每个字符对应的标签概率预测，需要取最大值
            batch_pred_labels = torch.argmax(batch_pred_labels, dim=-1)
        # 此时batch_true_labels和 batch_pred_labels的形状都为（batch_size, seq_len）
        for pred_labels, true_labels in zip(batch_pred_labels, batch_true_labels):
            # 将列表转为字符串
            # true_labels为张量，遍历的元素label也是张量，如果直接使用str(label)就会是“tensor(0)”形式的字符串
            # 所以必须先将张量转为int再转str：str(int(label))
            pred_labels = "".join(str(int(label)) for label in pred_labels)
            true_labels = "".join(str(int(label)) for label in true_labels)
            # 进行实体匹配，并记录到self.entity_info中
            self.matchEntity(pred_labels, true_labels)
        return

    """
    "O": 0,
    "B-PERSON": 1,
    "I-PERSON": 2,
    "B-LOCATION": 3,
    "I-LOCATION": 4,
    "B-ORGANIZATION": 5,
    "I-ORGANIZATION": 6,
    "B-TIME": 7,
    "I-TIME": 8
    """
    def matchEntity(self, pred_labels, true_labels):
        labels = {"PERSON": "12", "LOCATION": "34", "ORGANIZATION": "56", "TIME": "78"}
        for key, value in labels.items():
            pattern = f"({value})+"
            true_matches = re.finditer(pattern, true_labels)
            pred_matches = re.finditer(pattern, pred_labels)
            for true_match in true_matches:
                # re.finditer返回迭代器，不能使用len()
                self.entity_info[key]["total_count"] += 1   # 该类别实体总数
                entity_position = true_match.span()
                for pred_match in pred_matches:
                    self.entity_info[key]["predict_count"] += 1   # 该类别预测出的实体数
                    # 严格匹配：只有当实体和位置都正确才算预测正确
                    if pred_match.span() == entity_position:  # 索引元组
                        self.entity_info[key]["correct_count"] += 1   # 该类别预测正确的实体数
        return


    def calculateEntity(self):
        EPSON = 1e-5
        total_count = 0
        predict_count = 0
        correct_count = 0
        f1_all = []
        for category, entity_dict in self.entity_info.items():
            total_count += entity_dict["total_count"]
            predict_count += entity_dict["predict_count"]
            correct_count += entity_dict["correct_count"]
            # 宏观F1：计算每个类别的P/R/F1，然后对F1取平均
            if self.F1_type == 1:
                p = correct_count / (predict_count + EPSON)
                r = correct_count / (total_count + EPSON)
                f1 = 2 * p * r / (p + r + EPSON)
                f1_all.append(f1)
                print(f"类别-{category} 预测结果：准确率P={p}，召回率R={r}，F1={f1}")
        if self.F1_type == 1:
            print(f"各个标签类别平均F1={np.mean(f1_all)}")
            return
        # 微观F1：统计所有类别，一起计算P/R/F1
        p = correct_count / (predict_count + EPSON)
        r = correct_count / (total_count + EPSON)
        f1 = 2 * p * r / (p + r + EPSON)
        print(f"统计所有标签类别结果：准确率P={p}，召回率R={r}，F1={f1}")
        return






