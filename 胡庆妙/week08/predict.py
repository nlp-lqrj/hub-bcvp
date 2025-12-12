# -*- coding: utf-8 -*-
import torch
import jieba
from loader import load_data
from config import Config
from model import SiameseNetwork, choose_optimizer

"""
模型效果验证
"""


class Predictor:
    def __init__(self, config, model, knwb_data):
        self.config = config
        self.model = model
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()

        self.train_data = knwb_data
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema  # {标准问label: 标准问id}
        self.std_ques_id_to_label = dict((y, x) for x, y in self.schema.items())  # {标准问id: 标准问label}

        # 将知识库中的问题向量化，为匹配做准备，最终得到shape为[常用问的数量, embed_size] 的张量
        self.knwb_to_vector()

    # 将知识库中的问题向量化，为匹配做准备
    # 每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮验证都重新进行向量化
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

    def text_to_idvec(self, text):
        vec = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                vec.append(self.vocab.get(word, self.vocab["<unk>"]))
        else:
            for char in text:
                vec.append(self.vocab.get(char, self.vocab["<unk>"]))

        vec = vec[:self.config["sentence_len"]]  # 截断
        vec += [0] * (self.config["sentence_len"] - len(vec))  # 补齐
        return vec

    def predict(self, sentence):
        ques_idvec = self.text_to_idvec(sentence)  # [sen_len]
        ques_idvec = torch.LongTensor([ques_idvec])  # [1,sen_len]
        if torch.cuda.is_available():
            ques_idvec = ques_idvec.cuda()
        with torch.no_grad():
            ques_vector = self.model(ques_idvec)  # [1, sen_len] -> [1, embed_size]

        # ques_vector: [1, embed_size]
        # knwb_vectors: [常用问的数量, embed_size] -> [embed_size, 常用问的数量]
        # [1, embed_size] mm [embed_size, 常用问的数量] -> [1, 常用问的数量]
        res = torch.matmul(ques_vector, self.knwb_vectors.T)

        hit_ques_id = int(torch.argmax(res.squeeze()))  # 取得相似度最大的那个问题向量的编号
        std_ques_id = self.ques_id_to_std_ques_id[hit_ques_id]  # 转换成标准问的编号

        return self.std_ques_id_to_label[std_ques_id]  # 根据标准问id转换为标准问题


if __name__ == "__main__":
    # 加载训练数据(必须先加载训练数据，再加载模型，因为加载模型时依赖vocab_size)
    train_data = load_data(Config["train_data_path"], Config)  # 加载训练库，作为知识库

    model = SiameseNetwork(Config)
    model.load_state_dict(torch.load("model_output/epoch_10.pth"))  # 加载模型参数

    pd = Predictor(Config, model, train_data)
    while True:
        # sentence = "固定宽带服务密码修改"
        sentence = input("请输入问题：")
        res = pd.predict(sentence)
        print("\t> " + res)
