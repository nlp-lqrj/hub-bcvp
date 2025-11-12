# coding: utf-8

import math
import jieba
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

"""
 基于训练好的词向量模型进行聚类: 
 采用Kmeans算法, 对数千篇文章进行分类; 计算各簇的平均距离, 按平均距离排序, 优先输出平均距离最小的簇.
"""


def load_corpus(path):
    sentences = []  # like:  [['新增', '资金', '入场', '创', '年内', '新高'], ['记者', ...], ...]
    with open(path, encoding="utf8") as f:
        for line in f:
            sentences.append(jieba.lcut(line.strip()))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentence_to_vector(sentences, model):
    vectors = []
    for sentence in sentences:
        # print("words", words)
        vector = np.zeros(model.vector_size)
        # 对这句话所有的词向量相加求平均，作为句子的向量
        for word in sentence:
            try:
                vector += model.wv[word]  # 从词向量模型(model)中获取词"word"对应的向量表示
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(sentence))
    return np.array(vectors)


def main():
    sentences = load_corpus("word2vec_kmeans_titles.txt")  # 加载待分类的语料(文章标题)
    model = Word2Vec.load(r"word2vec.pth")  # 加载词向量模型, r 前缀表示这是一个原始字符串（raw string），可以防止路径中的反斜杠被转义
    vectors = sentence_to_vector(sentences, model)  # 将所有标题向量化

    # cluster_num = int(math.sqrt(len(sentences)))  # 指定聚类数量
    cluster_num = 35  # 通过肘部法则找到的最佳K值，参考：word2vec_kmeans_findbest.py
    print("指定聚类数量：", cluster_num)

    kmeans = KMeans(cluster_num)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    label_sentences_dict = defaultdict(list)  # {簇号: [句子1, 句子2, ...]}, defaultdict(list) 表示当访问不存在的键时，会自动创建一个空列表[]作为默认值
    label_sum_distance = {i: 0 for i in range(cluster_num)}  # {簇号: 簇内向量与质心的距离之和}
    for label, sentence, vector in zip(kmeans.labels_, sentences, vectors):  # 取出簇号及对应的句子、向量
        label_sentences_dict[label].append(sentence)  # 将同簇号的标题放到一起
        center_point = kmeans.cluster_centers_[label]  # 簇号为label的质心
        label_sum_distance[label] += np.linalg.norm(np.array(vector) - center_point)  # 计算当前向量与质心的距离，并累加到簇内距离之和

    # 计算簇内平均距离, {簇号: 簇内平均距离}
    label_avg_distance = {}
    for label, sum_distance in label_sum_distance.items():
        # 簇内平均距离 = 簇内向量与质心的距离之和 / 簇内向量数量
        label_avg_distance[label] = sum_distance / np.sum(kmeans.labels_ == label)

    # 按簇内平均距离从小到大排序
    label_avg_distance = dict(sorted(label_avg_distance.items(), key=lambda x: x[1]))

    # 输出簇号、平均距离、簇内句子
    for label, avg_distance in label_avg_distance.items():
        sentence_list = label_sentences_dict[label]
        print(f"cluster: {label},  avg_distance: {avg_distance:.5f}")
        for i in range(min(10, len(sentence_list))):  # 随便打印几个，太多了看不过来
            print(''.join(sentence_list[i]))
        print("---------")


if __name__ == "__main__":
    main()
