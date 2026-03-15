


# 第六周作业：1.实现透视变化 2.实现kmeans



#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import cv2
from sklearn.cluster import KMeans
from transformers import BertModel

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

week2的例子，修改引入bert
添加透视变换和K-means实现
"""

class PerspectiveTransformer:
    """透视变换实现"""
    def __init__(self):
        pass
    
    def four_point_transform(self, image, pts):
        """
        执行四点透视变换
        Args:
            image: 输入图像
            pts: 四个点的坐标，顺序为 [左上, 右上, 右下, 左下]
        Returns:
            warped: 变换后的图像
        """
        # 解包点坐标
        (tl, tr, br, bl) = pts
        
        # 计算新图像的宽度
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # 计算新图像的高度
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # 构造目标点
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def demo_perspective_transform(self):
        """演示透视变换"""
        # 创建一个简单的测试图像
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (250, 150), (255, 255, 255), -1)
        cv2.putText(image, 'Test Image', (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 定义原始点（模拟透视效果）
        pts = np.array([
            [45, 45],    # 左上
            [255, 35],   # 右上  
            [265, 155],  # 右下
            [35, 165]    # 左下
        ], dtype="float32")
        
        # 执行透视变换
        warped = self.four_point_transform(image, pts)
        
        return image, warped


class CustomKMeans:
    """自定义K-means实现"""
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        """训练K-means模型"""
        np.random.seed(self.random_state)
        
        # 随机初始化质心
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for iteration in range(self.max_iters):
            # 分配样本到最近的质心
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # 更新质心
            new_centroids = np.array([X[self.labels == i].mean(axis=0) 
                                    for i in range(self.n_clusters)])
            
            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
            
        return self
    
    def predict(self, X):
        """预测样本所属的簇"""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def _compute_distances(self, X):
        """计算样本到所有质心的距离"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances
    
    def demo_kmeans(self, n_samples=100):
        """演示K-means聚类"""
        # 生成测试数据
        np.random.seed(42)
        X1 = np.random.normal([2, 2], 0.5, [n_samples//3, 2])
        X2 = np.random.normal([-2, -2], 0.5, [n_samples//3, 2])
        X3 = np.random.normal([0, -2], 0.5, [n_samples//3, 2])
        X = np.vstack([X1, X2, X3])
        
        # 使用自定义K-means
        kmeans_custom = CustomKMeans(n_clusters=3)
        kmeans_custom.fit(X)
        custom_labels = kmeans_custom.labels
        
        # 使用sklearn的K-means进行比较
        kmeans_sklearn = KMeans(n_clusters=3, random_state=42)
        sklearn_labels = kmeans_sklearn.fit_predict(X)
        
        return X, custom_labels, sklearn_labels, kmeans_custom.centroids


class TorchModel(nn.Module):
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(input_dim, 3)
        self.activation = torch.sigmoid
        self.dropout = nn.Dropout(0.5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        sequence_output, pooler_output = self.bert(x)
        x = self.classify(pooler_output)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)+1
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set("abc") & set(x) and not set("xyz") & set(x):
        y = 0
    elif not set("abc") & set(x) and set("xyz") & set(x):
        y = 1
    else:
        y = 2
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sample_length):
    model.eval()
    total = 200
    x, y = build_dataset(total, vocab, sample_length)
    y = y.squeeze()
    print("A类样本数量：%d, B类样本数量：%d, C类样本数量：%d"%(y.tolist().count(0), y.tolist().count(1), y.tolist().count(2)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d / %d, 正确率：%f"%(correct, total, correct/(correct+wrong)))
    return correct/(correct+wrong)


def demo_computer_vision():
    """演示计算机视觉功能：透视变换和K-means"""
    print("=" * 50)
    print("计算机视觉功能演示")
    print("=" * 50)
    
    # 1. 透视变换演示
    print("\n1. 透视变换演示")
    transformer = PerspectiveTransformer()
    original, transformed = transformer.demo_perspective_transform()
    print(f"原始图像形状: {original.shape}")
    print(f"变换后图像形状: {transformed.shape}")
    
    # 2. K-means演示
    print("\n2. K-means聚类演示")
    kmeans_demo = CustomKMeans()
    X, custom_labels, sklearn_labels, centroids = kmeans_demo.demo_kmeans(150)
    print(f"数据点形状: {X.shape}")
    print(f"自定义K-means质心: \n{centroids}")
    print(f"自定义K-means标签分布: {np.bincount(custom_labels)}")
    print(f"Sklearn K-means标签分布: {np.bincount(sklearn_labels)}")
    
    # 3. 在NLP特征上应用K-means的示例
    print("\n3. 在NLP特征上应用K-means的示例")
    # 模拟一些文本特征
    nlp_features = np.random.randn(100, 10)  # 100个样本，10维特征
    nlp_kmeans = CustomKMeans(n_clusters=3)
    nlp_kmeans.fit(nlp_features)
    nlp_clusters = nlp_kmeans.labels
    print(f"NLP特征聚类结果分布: {np.bincount(nlp_clusters)}")
    
    return original, transformed, X, custom_labels, centroids


def main():
    # 演示计算机视觉功能
    demo_computer_vision()
    
    print("\n" + "=" * 50)
    print("开始NLP模型训练")
    print("=" * 50)
    
    # NLP模型训练
    epoch_num = 15
    batch_size = 20
    train_sample = 1000
    char_dim = 768
    sentence_length = 6
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    log = []
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    
    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 768
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print(int(torch.argmax(result[i])), input_string, result[i])


if __name__ == "__main__":
    main()
    test_strings = ["juvaee", "yrwfrg", "rtweqg", "nlhdww"]
    predict("model.pth", "vocab.json", test_strings)