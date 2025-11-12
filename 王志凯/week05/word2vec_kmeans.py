"""
使用word2vec和kmeans对样本进行分类
要求：对分类样本按照簇内平均距离进行排序
"""

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import re
import jieba
import math


# 词向量模型
class Word2VecModel():
    def __init__(self, corpus_path, train=True):
        self.corpus_path = corpus_path
        self.train = train  # 是否训练模型（如果训练过，就传False，直接加载训练好的模型）
        self.model = None
        self.sentences_words = defaultdict(list)  # 文本对应分好词的列表
        self._train_word2vec()

    def _train_word2vec(self):
        # 使用jieba分词
        sentences_cut = self.__corpus_to_sentences()
        if self.train:
            self.model = Word2Vec(sentences_cut, sg=1)
            self.model.save("word2vec_model.bin")
        else:
            # 如果为False，表示已经训练过，只需要加载模型
            self.model = Word2Vec.load("word2vec_model.bin")

    # 将文档内容转化成文本列表并分词
    def __corpus_to_sentences(self):
        with open(self.corpus_path, encoding="utf8") as f:
            sentences = []
            for line in f:
                sentence = line.strip()
                words = jieba.lcut(sentence)
                self.sentences_words[sentence] = words
                sentences.append(words)
            return sentences


# 使用kmeans进行聚类
class KMeansCluster():
    def __init__(self, sentences, n_clusters):
        self.sentences = sentences
        self.n_clusters = n_clusters
        self.lables = None
        self.centers = None
        self.__cluster()

    def __cluster(self):
        kmeans = KMeans(self.n_clusters)
        kmeans.fit(self.sentences)
        self.labels = kmeans.labels_
        self.centers = kmeans.cluster_centers_


# 将文本转化为向量
def sentences_to_vectors(model, sentences_words):
    sentences_vectors = defaultdict(list)
    for sentence, words in sentences_words.items():
        vectors = []
        for word in words:
            try:
                vectors.append(model.wv[word])
            except KeyError:
                # 未出现的词用全0向量代替（不贡献文本向量）
                vectors.append(np.zeros(model.vector_size))
        # 文本中所有词向量加和求平均得到该文本向量
        sentences_vectors[sentence] = list(np.array(vectors).mean(axis=0))
    return sentences_vectors


def samples_sort(sentences_vectors, labels, centers, sort_type=0):
    labels_sentences = defaultdict(list)
    labels_vectors = defaultdict(list)
    for sent, vector, centroid_idx in zip(sentences_vectors.keys(), sentences_vectors.values(), labels):
        labels_sentences[centroid_idx].append(sent)
        labels_vectors[centroid_idx].append(vector)

    if sort_type == 0:
        # 按样本数降序排序
        return {k: v for k, v in sorted(labels_sentences.items(), key=lambda x: len(x[1]), reverse=True)}
    elif sort_type == 1:
        # 按样本到质心的平均距离升序排序
        labels_mean_dis = defaultdict(float)
        for i, centroid in enumerate(centers):
            vectors = labels_vectors[i]
            # 所有样本到质心的距离
            distance = 0
            for vector in vectors:
                # 单个样本到质心的距离（欧式距离）
                dis = 0
                # 分别计算每一个维度的距离
                for dim in range(len(vector)):
                    dis += (vector[dim] - centroid[dim]) ** 2
                distance += dis ** 0.5
            labels_mean_dis[i] = distance / len(vectors)
        # 根据labels_mean_dis中值的大小对labels_sentences进行排序
        return {k: labels_sentences[k] for k in sorted(labels_sentences, key=lambda x: labels_mean_dis[x])}


if __name__ == "__main__":
    word2vec = Word2VecModel("titles.txt")
    # 如果已经训练过模型，则将train=False，只需要加载模型
    # word2vec = Word2VecModel("titles.txt", train=False)
    model = word2vec.model
    # 文本对应文本分词
    sentences_words = word2vec.sentences_words
    # 文本对应文本向量
    sentences_vectors = sentences_to_vectors(model, sentences_words)

    # 分簇（簇类数量=样本数开方）
    n_clusters = int(math.sqrt(len(sentences_vectors)))
    kmeans = KMeansCluster(list(sentences_vectors.values()), n_clusters)
    labels = kmeans.labels  # 所有样本标签（每个样本所属的质心索引）
    centers = kmeans.centers  # 所有质心的坐标（向量）

    # sort_type=0:按照簇内样本数降序排序
    # sort_type=1:按照簇内所有样本到质心的平均距离升序排序
    label_sentences = samples_sort(sentences_vectors, labels, centers, sort_type=1)
    for label, sentences in label_sentences.items():
        print(f"簇编号：{label}")
        count = 10 if len(sentences) > 10 else len(sentences)
        for i in range(count):
            print(sentences[i])
        print("=" * 50)