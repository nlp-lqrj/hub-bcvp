# coding: utf-8

import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""
 采用Kmeans对数千篇文章进行分类, 计算在不同簇数时所有向量与其质心的平均距离, 绘制簇数与平均距离的趋势图, 运用肘部法则找出最佳分类. 
 肘部法则: 
    当簇数K 小于真实簇数时，增加 K 会极大地增加每个簇的紧凑性（平均距离大幅下降）。
    当簇数K 达到或超过真实簇数时，再增加 K，平均距离的下降幅度会突然变得很平缓。这个拐点就是“肘部”。
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


def classify(vectors, cluster_num):
    # print("指定聚类数量：", cluster_num)
    kmeans = KMeans(cluster_num)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 所有向量与质心的距离之和
    sum_distance = 0.0
    for label, vector in zip(kmeans.labels_, vectors):  # 取出簇号及对应的向量
        center_point = kmeans.cluster_centers_[label]  # 簇号为label的质心
        sum_distance += np.linalg.norm(np.array(vector) - center_point)  # 计算当前向量与质心的距离，并累加到距离之和

    # 所有向量与对应质心的平均距离
    avg_distance = sum_distance / len(vectors)
    print(f"指定聚类数量为 {cluster_num} 时, 平均距离为: {avg_distance}")
    return avg_distance


def main():
    sentences = load_corpus("word2vec_kmeans_titles.txt")  # 加载待分类的语料(文章标题)
    model = Word2Vec.load(r"word2vec.pth")  # 加载词向量模型
    vectors = sentence_to_vector(sentences, model)  # 将所有标题向量化

    kk = []
    dis = []
    # 指定聚类数量，计算所有向量与对应质心的平均距离
    for k in range(2, int(math.sqrt(len(sentences)))*4):
        avg_distance = classify(vectors, k)
        kk.append(k)
        dis.append(avg_distance)

    print("\n绘制簇数与平均距离的趋势图...")
    plt.plot(kk, dis, label="平均距离")
    plt.xlabel("簇数(K值)")
    plt.ylabel("平均距离")
    plt.legend(title='Kmeans-K值与平均距离趋势图')

    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统下中文可以使用'SimHei'或'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.show()
    return


if __name__ == "__main__":
    main()
