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
    sentences = set()   # 做KMeans聚类任务前，先做文本去重处理
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


#计算两个向量的欧式距离
def get_distance(d1, d2):
    # tmp = 0
    # for i in range(len(p1)):
    #     tmp += pow(d1[i] - d2[i], 2)
    tmp = sum(pow(x - y, 2) for x, y in zip(d1, d2))
    return pow(tmp, 0.5)

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
        #kmeans.cluster_centers_ #每个聚类中心
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        # index = sentences.index(sentence)
        # sentence_vector = vectors[index]

    # 1. 计算label类内平均距离
    label_distance_dict = defaultdict(float)
    for label, sentences in sentence_label_dict.items():
        center = kmeans.cluster_centers_[label]
        sentence_vectors = sentences_to_vectors(sentences, model)
        distance = 0
        for sentence_vector in sentence_vectors:
            distance += get_distance(sentence_vector, center)
        label_distance_dict[label] = distance / len(sentences)

    # 2. 按距离从小到大排序
    sorted_label_distance_list = sorted(label_distance_dict.items(), key=lambda x: x[1])  # list，排序后的键值对元组（label, distance）

    # 3. 按排序打印聚类结果
    for label, distance in sorted_label_distance_list:
        print("============= cluster %s ===============" % label)
        print("=> 类内平均距离：%.4f" % distance)
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        # print("---------")

if __name__ == "__main__":
    main()

