# -*- coding: utf-8 -*-
import re
import torch
from transformers import BertTokenizer

from config import Config
from model import setup_rola_model
import loader

"""
模型的测试、应用
"""


class Predictor:
    def __init__(self, config, model_param_filename):
        self.config = config
        self.model_param_filename = model_param_filename
        self.schema = loader.load_schema(self.config["schema_path"])
        self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])

        self.model = self.load_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def load_model(self):
        # 支持RoLA的模型
        model = setup_rola_model(self.config["pretrain_model_path"], len(self.schema))
        state_dict = model.state_dict()
        # print("base model params: ")
        # for key, value in state_dict.items():
        #     print("\t\t> ", key, value.shape)

        # 加载微调部分的参数
        loaded_weight = torch.load(self.model_param_filename)
        print("loaded_weight.keys:", loaded_weight.keys())
        for key, value in loaded_weight.items():
            print("\t\t> ", key, value.shape)

        # 更新参数并重新加载到模型
        state_dict.update(loaded_weight)
        model.load_state_dict(state_dict)
        return model

    def predict(self, sentence):
        input_ids = loader.encode_sentence(self.tokenizer, sentence, truncate=False, padding=False)
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # [1, sentence_len]
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        with torch.no_grad():
            logits = self.model(input_ids)[0]  # [1, sen_len, num_label]
            pred_label_ids = torch.argmax(logits, dim=-1)  # [1, sen_len]
            pred_label_ids = pred_label_ids[0].tolist()  # [sen_len]

        # 在本文中标记预测出来的实体，输出标记后的文本
        return self.label_entities_in_sentence(sentence, pred_label_ids)

    # 在本文中标注预测出来的实体，输出标注后的文本
    @staticmethod
    def label_entities_in_sentence(sentence, label_ids):
        """
        Args:
            sentence: 如： "他是彭德怀"
            label_ids: 如：[8, 8, 2, 6, 6]
        Returns:
            标注后的本文: 如：他是{彭德怀/PERSON}
        """
        labels = "".join([str(x) for x in label_ids])
        output = ""
        i = 0
        for it in re.finditer("(04+)|(15+)|(26+)|(37+)", labels):
            item = it.group()  # 获取匹配的字符串
            s, e = it.span()  # 匹配到的子字符串的起始和结束位置
            if re.fullmatch(r'(04+)', item):
                output += sentence[i:s] + " {" + sentence[s:e] + "/LOCATION} "
            elif re.fullmatch(r'(15+)', item):
                output += sentence[i:s] + " {" + sentence[s:e] + "/ORGANIZATION} "
            elif re.fullmatch(r'(26+)', item):
                output += sentence[i:s] + " {" + sentence[s:e] + "/PERSON} "
            elif re.fullmatch(r'(37+)', item):
                output += sentence[i:s] + " {" + sentence[s:e] + "/TIME} "
            i = e
        if i < len(sentence):
            output += sentence[i:]
        return output


if __name__ == "__main__":
    pd = Predictor(Config, "output/ner_LoRA.pth")

    input_text = ("建设海南自由贸易港的战略目标，就是要把海南自由贸易港打造成为引领我国新时代对外开放的重要门户。"
                  "2025年11月6日，习近平总书记在海南省三亚市听取海南自由贸易港建设工作汇报并发表重要讲话。")

    print(">>", input_text)
    res = pd.predict(input_text)
    print("<<", res)
    print("------------------------------------")

    test_data = loader.load_data(Config["valid_data_path"], Config, shuffle=False)
    for sen in test_data.dataset.sentences[:3]:
        print(">>", sen)
        res = pd.predict(sen)
        print("<<", res)
        print("------------------------------------")
