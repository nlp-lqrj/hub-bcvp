import numpy as np
import math
import json
import re
import jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 加载Word2Vec模型
def load_w2v_model(model_path: str) -> Word2Vec:
    """加载预训练的Word2Vec模型"""
    return Word2Vec.load(model_path)


# 加载并预处理语料
def load_corpus(corpus_path: str) -> set:
    """加载句子语料，去重并分词"""
    unique_sentences = set()
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            raw_sent = line.strip()
            if raw_sent:
                segmented = " ".join(jieba.cut(raw_sent))
                unique_sentences.add(segmented)
    print(f"加载去重后语料条数：{len(unique_sentences)}")
    return unique_sentences


# 句子转向量
def corpus_to_vecs(corpus: set, w2v_model: Word2Vec) -> np.ndarray:
    """将分词后的句子转换为平均词向量"""
    vec_dim = w2v_model.vector_size
    sentence_vecs = []

    for sent in corpus:
        words = sent.split()
        avg_vec = np.zeros(vec_dim, dtype=np.float32)
        valid_word_cnt = 0

        for word in words:
            if word in w2v_model.wv:
                avg_vec += w2v_model.wv[word]
                valid_word_cnt += 1

        # 避免除数为0
        if valid_word_cnt > 0:
            avg_vec /= valid_word_cnt
        sentence_vecs.append(avg_vec)

    return np.array(sentence_vecs)


# 计算两点欧氏距离
def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个向量的欧氏距离"""
    return np.linalg.norm(vec1 - vec2)


# 计算簇内平均距离（优化循环结构）
def cluster_avg_dist(cluster_vecs: np.ndarray, centroid: np.ndarray) -> float:
    """计算单个聚类簇的平均距离（簇内所有点到质心的平均）"""
    total_dist = sum(euclidean_distance(vec, centroid) for vec in cluster_vecs)
    return total_dist / len(cluster_vecs)


# 按簇内距离排序聚类结果（调整字典构建顺序）
def sort_clusters_by_dist(cluster_dict: dict, centroids: np.ndarray, w2v_model: Word2Vec) -> dict:
    """根据簇内平均距离升序排序聚类结果"""
    # 计算每个簇的平均距离
    dist_map = {}
    for label, sentences in cluster_dict.items():
        cluster_vecs = corpus_to_vecs(sentences, w2v_model)
        dist_map[label] = cluster_avg_dist(cluster_vecs, centroids[label])

    # 按距离排序并重构聚类字典
    sorted_labels = sorted(dist_map.keys(), key=lambda x: dist_map[x])
    sorted_cluster = {label: cluster_dict[label] for label in sorted_labels}
    return sorted_cluster

def main():
    # 配置路径
    W2V_MODEL_PATH = r'model.w2v'
    CORPUS_PATH = r'titles.txt'

    # 1. 加载依赖资源
    w2v_model = load_w2v_model(W2V_MODEL_PATH)
    corpus = load_corpus(CORPUS_PATH)

    # 2. 句子转向量
    sent_vecs = corpus_to_vecs(corpus, w2v_model)

    # 3. KMeans聚类
    cluster_num = int(math.sqrt(len(corpus)))
    print(f"聚类簇数自动设置为：{cluster_num}")
    kmeans = KMeans(n_clusters=cluster_num, random_state=42)
    kmeans.fit(sent_vecs)

    # 4. 构建聚类结果字典
    cluster_result = defaultdict(list)
    for sent, label in zip(corpus, kmeans.labels_):
        cluster_result[label].append(sent)

    # 5. 打印原始聚类结果
    print("\n=== 原始聚类结果（每个簇显示前5条）===")
    for label, sents in cluster_result.items():
        print(f"\n簇 {label}（共{len(sents)}条）：")
        for idx, sent in enumerate(sents[:5]):
            print(f"  {idx + 1}. {sent.replace(' ', '')}")
        print("-" * 50)

    # 6. 打印排序后的聚类结果
    print("\n=== 按簇内平均距离排序后的结果 ===")
    sorted_clusters = sort_clusters_by_dist(cluster_result, kmeans.cluster_centers_, w2v_model)
    for label, sents in sorted_clusters.items():
        print(f"\n簇 {label}（共{len(sents)}条）：")
        for idx, sent in enumerate(sents[:5]):
            print(f"  {idx + 1}. {sent.replace(' ', '')}")
        print("-" * 50)


if __name__ == "__main__":
    main()