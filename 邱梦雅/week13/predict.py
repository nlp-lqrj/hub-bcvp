# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
import logging
from collections import defaultdict
from config import Config
from model import TorchModel
from transformers import BertTokenizer
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig
"""
模型效果测试
"""

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
log_path = "predict.log"
handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
logger.addHandler(handler)

class SentenceLabel:
    def __init__(self, config):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])  # "bert_path": r"D:\pretrain_models\bert-base-chinese"
        # self.vocab = self.load_vocab(config["vocab_path"])
        self.vocab = self.tokenizer.vocab
        self.load_model_with_peft(config)

    def load_model_with_peft(self, config):
        # 大模型微调策略
        self.tuning_tactics = config["tuning_tactics"]
        logger.info("正在使用 %s" % self.tuning_tactics)
        tuning_tactics = config["tuning_tactics"]
        if tuning_tactics == "lora_tuning":
            peft_config = LoraConfig(
                r=8,
                lora_alpha=32,  # 缩放因子
                lora_dropout=0.1,
                target_modules=["query", "key", "value"]  # 在 q、k、v 线性层添加LoRA模块 bert模型中每一个网络层是有名字的 state_dict
            )
        elif tuning_tactics == "prompt_tuning":
            peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        elif tuning_tactics == "prefix_tuning":
            peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        elif tuning_tactics == "p_tuning":
            peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        # 重建模型
        model = TorchModel(config)
        # print(model.state_dict().keys())
        # print("====================")
        self.model = get_peft_model(model, peft_config)
        # print(self.model.state_dict().keys())
        # print("====================")

        state_dict = self.model.state_dict()
        # 将微调部分权重加载
        if tuning_tactics == "lora_tuning":
            loaded_weight = torch.load('model_output/lora_tuning_epoch_15.pth')
        elif tuning_tactics == "prompt_tuning":
            loaded_weight = torch.load('model_output/prompt_tuning_epoch_15.pth')
        elif tuning_tactics == "prefix_tuning":
            loaded_weight = torch.load('model_output/prefix_tuning_epoch_15.pth')
        elif tuning_tactics == "p_tuning":
            loaded_weight = torch.load('model_output/p_tuning_epoch_15.pth')

        # print(loaded_weight.keys())
        for key, value in loaded_weight.items():
            logger.info(f"{key} : {value.shape}")
        state_dict.update(loaded_weight)

        # 权重更新后重新加载到模型
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index  # bert词表vocab.txt中第一行就是[PAD]，索引为0
            self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))[0]  # => (batch_size, seq_len)
        if not self.config["use_crf"]:
            res = torch.argmax(res, dim=-1)         # => (batch_size, seq_len)
            res = res.cpu().detach().tolist()

        pred_entities = self.decode(sentence, res)
        logger.info(f"pred_labels: {res}")
        for key, value in pred_entities.items():
            logger.info(f"{key} : {value}")
        return pred_entities

    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''
    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results


if __name__ == "__main__":
    sl = SentenceLabel(Config)

    sentence = "哈总统纳扎尔巴耶夫最近还突然提起,俄因租赁拜科努尔火箭发射基地,对哈欠有4.5亿美元债务,等等。"
    res = sl.predict(sentence)
    logger.info(res)
    logger.info("==============")

    sentence = "二十六日,法国总统希拉克、俄罗斯总统叶利钦和德国总理科尔(自左至右)在莫斯科举行会晤。"
    res = sl.predict(sentence)
    logger.info(res)
    logger.info("==============")

    sentence = "法国芭蕾又登北京舞台南锡芭蕾舞团明晚演《天鹅湖》中国人民银行行长戴相龙作了题为“亚洲金融危机与我们的对策”的报告;国家发展计划委员会主任曾培炎作了“当前的经济形势与任务”的报告;中央党校常务副校长郑必坚作了“十五大精神与邓小平理论”的报告;中央统战部副部长刘延东作了“统一战线和多党合作”的报告……实际上,2000年问题的暴露将会比人们预计的要早。"
    res = sl.predict(sentence)
    logger.info(res)