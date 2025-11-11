# 基于训练好的词向量进行聚类
import math 
import re 
import json 
import jieba 
import numpy as np 
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model 

def load_sentence(path):
    sentences = set()
    with open(path,encoding='utf8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：",len(sentences))
    return sentences

def sentences_to_vectors(sentences,model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector/len(words))
    return np.array(vectors)

def sort_clusters(clusters:dict,centers,model)->dict:
    # clusters:{label:[sentences]},
    cluster_avg_distance = defaultdict(int)
    for label,sentences in clusters.items():
        vectors = sentences_to_vectors(sentences,model)
        cluster_avg_distance[label] = calculate_cluster_distance(vectors,centers[label])
    sorted_cluster_avg_distance=dict(sorted(cluster_avg_distance.items(),key=lambda x :x[1]))
    sorted_cluster = dict()
    for label in sorted_cluster_avg_distance.keys():
        sorted_cluster[label] = clusters[label]
    return sorted_cluster

def calculate_cluster_distance(vectors,center_point):
    sum_ = 0
    for vector in vectors:
        sum_ += calculate_distance(vector,center_point)
    return sum_/len(vectors)
        
def calculate_distance(p1,p2):
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i]-p2[i],2)
    return math.sqrt(tmp)

def main():
    model_path = r'model.w2v'
    sentence_path = r'titles.txt'
    model = load_word2vec_model(model_path)
    sentences = load_sentence(sentence_path)
    vectors = sentences_to_vectors(sentences,model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("制定聚类数量为：",n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    for sentence,label in zip(sentences,kmeans.labels_):
        sentence_label_dict[label].append(sentence)
    for label,sentences in sentence_label_dict.items():
        print(f"cluster{label}:")
        for i in range(min(5,len(sentences))):
            print(sentences[i].replace(" ",""))
        print("-------------")
    
    print("根据簇内平均距离排序后：")
    sorted_clusters = sort_clusters(sentence_label_dict,kmeans.cluster_centers_,model)
    for label,sentences in sorted_clusters.items():
        print(f"cluster{label}:")
        for i in range(min(5,len(sentences))):
            print(sentences[i].replace(" ",""))
        print("-------------")

if __name__=="__main__":
    main()
