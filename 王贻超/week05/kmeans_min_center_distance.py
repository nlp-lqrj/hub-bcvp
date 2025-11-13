#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

# 计算每个聚类的平均中心距离
def calculate_cluster_distances(kmeans, vectors, labels):
    cluster_distances = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        # 获取属于该聚类的所有向量
        cluster_vectors = vectors[labels == i]
        # 计算每个向量到聚类中心的距离
        distances = [np.linalg.norm(vec - center) for vec in cluster_vectors]
        # 计算平均距离
        avg_distance = np.mean(distances) if len(distances) > 0 else 0
        cluster_distances[i] = avg_distance
    return cluster_distances

def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    
    # 计算每个聚类的中心距
    cluster_distances = calculate_cluster_distances(kmeans, vectors, kmeans.labels_)
    
    # 按照中心距排序，距离小的在前面
    sorted_clusters = sorted(cluster_distances.items(), key=lambda x: x[1])
    
    # 按排序后的顺序输出聚类结果
    for label, avg_distance in sorted_clusters:
        sentences = sentence_label_dict[label]
        print("cluster %s : 平均中心距离=%.4f" % (label, avg_distance))
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
