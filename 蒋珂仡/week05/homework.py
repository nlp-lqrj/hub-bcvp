#!/usr/bin/env python3
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re  #导入正则表达式
import json  #导入json库
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


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    sentence_list=list(sentences)
    vectors = sentences_to_vectors(sentence_list, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    labels=kmeans.labels_
    cluster_centers=kmeans.cluster_centers_
    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)

    for i in range(len(labels)):
        sentence=sentence_list[i]
        label = labels[i]
        vector=vectors[i]

        sentence_label_dict[label].append(sentence)
        vector_label_dict[label].append(vector)

    cluster_metric=[]
    for label,vecs_in_cluster in vector_label_dict.items():
        center_vec=cluster_centers[label]
        total_distance=0
        for vec in vecs_in_cluster:
            total_distance += np.linalg.norm(vec-center_vec)
        average_distance=total_distance/len(vecs_in_cluster)
        cluster_metric.append({"label":label,"average_distance":average_distance,"sentences":sentence_label_dict[label]})

    cluster_metric.sort(key=lambda x: x["average_distance"])
    print("按照平均距离由小到大进行排列")
    for cluster in cluster_metric:
        label=cluster["label"]
        average_distance=cluster["average_distance"]
        sentences=cluster["sentences"]


        print(f"\ncluster {label} (平均距离: {average_distance:.4f}) :")

        for i in range(min(10, len(sentences))):  # 随便打印几个
            print(sentences[i].replace(" ", ""))
        print("---------")







    #
    # sentence_label_dict = defaultdict(list)
    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     #kmeans.cluester_centers_ #每个聚类中心
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")



if __name__ == "__main__":
    main()

