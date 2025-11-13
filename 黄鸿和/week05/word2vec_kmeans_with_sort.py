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
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
        if not words:
            continue
        vector = np.zeros(model.vector_size)
        #把一句话划分的 所有词的向量 相加求平均，作为句子向量
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
    sentences_list = list(sentences)  # 固定顺序，保证与 vectors/labels 对齐
    vectors = sentences_to_vectors(sentences_list, model)   #将所有标题向量化
    # print('sentences_list', len(sentences_list))
    # print('vectors', vectors.shape)

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    # print("111111111111111111111111111111111111")
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    # print("kmeans", kmeans)
    # print("vectors", vectors)
    try:
        kmeans.fit(vectors)          #进行聚类计算
    except Exception as e:
        print("kmeans.fit 发生异常：", e)
    # print("111111111111111111111111111111111111")

    # 统计并打印簇标签分布（数量与占比）
    if hasattr(kmeans, "labels_"):
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        print("簇标签分布：")
        for l, c in zip(unique, counts):
            print("label=%s  count=%s  ratio=%.2f%%" % (l, c, 100.0 * c / total))
    else:
        print("未获取到 labels_，可能是 kmeans.fit 失败。")
        return

    # # 原有结果：将同标签的句子放到一起并打印
    # sentence_label_dict = defaultdict(list)
    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     #kmeans.cluester_centers_ #每个聚类中心 
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
        
    # for label, sents in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sents))):  #随便打印几个，太多了看不过来
    #         print(sents[i].replace(" ", ""))
    #     print("---------")

    # ============== 第五周作业需求：基于kmeans结果“类内距离”的排序输出 ==============
    # 1.思路：先找到kmeans框架中,代表质点中心的变量（cluster_centers_）
    # 2.然后计算每条句子向量与其所属簇中心的欧氏距离（句子向量是每个分词得到的向量，然后做一个 均值求和的操作得到的）
    # 3.按距离从小到大排序后再打印
    print("按类内距离排序后的结果：")
    centers = kmeans.cluster_centers_
    # sentences_list 已经跟 vectors 进行对其了，vectors的 label 会跟 sentences_list 对应
    distance_sorted = defaultdict(list)
    for idx, (sent, label) in enumerate(zip(sentences_list, kmeans.labels_)):
        vec = vectors[idx]
        center = centers[label]
        # 记录每个 sent 的 vec 与 label对应centers的  欧氏距离
        # 两向量差的范数，即欧氏距离，距离越小表示越“靠近”簇中心
        dist = float(np.linalg.norm(vec - center))
        # 得到每个句子与其 类内中心 的距离
        distance_sorted[label].append((dist, sent))
    # 根据label, 开始进行 各簇内部按距离升序
    out_path = "kmeans_sorted.txt"
    fout = open(out_path, "w", encoding="utf8")
    for label in sorted(distance_sorted.keys()):
        pairs = distance_sorted[label]
        pairs.sort(key=lambda x: x[0])
        print("cluster %s (sorted by distance):" % label)
        fout.write("cluster %s (sorted by distance):\n" % label)
        for dist, sent in pairs[:10]:  # 仍然每簇最多打印10条
            print("%s\t\t距离: %.4f" % (sent.replace(" ", ""), dist))
            fout.write("%s\t\t距离: %.4f\n" % (sent.replace(" ", ""), dist))
        print("---------")
        fout.write("---------\n")
    # ======================================================================
    fout.close()

if __name__ == "__main__":
    main()


