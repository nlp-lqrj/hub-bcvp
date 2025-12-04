import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def get_length_distribution(data_dict, title):
    """获取文本长度分布并绘制直方图"""
    lengths = [len(record["review"]) for record in data_dict]

    # 打印统计信息
    print(f"\n{title}统计信息:")
    print(f"最大长度: {max(lengths)}")
    print(f"最小长度: {min(lengths)}")
    print(f"平均长度: {np.mean(lengths):.2f}")
    print(f"中位数长度: {np.median(lengths):.2f}")
    print(f"标准差: {np.std(lengths):.2f}")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'{title}文本长度分布')
    plt.xlabel('文本长度')
    plt.ylabel('频次')
    plt.grid(True, alpha=0.3)
    plt.show()

    # 打印长度区间统计
    percentiles = [10, 25, 50, 75, 90, 95, 97, 99]
    print(f"\n{title}长度分位数统计:")
    for p in percentiles:
        print(f"{p}%分位数: {np.percentile(lengths, p):.2f}")

    return lengths


def get_detailed_label_stats(data_dict, title):
    """获取更详细的标签统计信息"""
    stats = {
        'total_samples': len(data_dict),
        'label_counts': {},
        'label_percentages': {}
    }

    for record in data_dict:
        label = record['label']
        stats['label_counts'][label] = stats['label_counts'].get(label, 0) + 1

    for label, count in stats['label_counts'].items():
        stats['label_percentages'][label] = (count / stats['total_samples']) * 100

    print(f"\n{title}详细统计:")
    print(f"总样本数: {stats['total_samples']}")
    for label in stats['label_counts']:
        print(f"{label}: {stats['label_counts'][label]} ({stats['label_percentages'][label]:.2f}%)")

    return stats


def get_max_len(train_dict, valid_dict):
    max_len = 0
    max_len_seq = ""
    for record in train_dict + valid_dict:
        review = record["review"]
        if len(review) > max_len:
            max_len = len(review)
            max_len_seq = review
    return max_len, max_len_seq

def get_avg_len(train_dict, valid_dict):
    total_len = 0
    for record in train_dict + valid_dict:
        review = record["review"]
        total_len += len(review)
    avg_len = total_len / (len(train_dict) + len(valid_dict))
    return avg_len


if __name__ == "__main__":
    # 读取CSV文件
    file_path = '文本分类练习.csv'  # 文件在当前目录
    df = pd.read_csv(file_path)

    # 划分数据集为训练集和验证集（80%训练，20%验证）
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    # 获取训练集和验证集的索引（假设CSV文件有索引列）
    # train_indices = train_df.index.tolist()
    # valid_indices = valid_df.index.tolist()

    # 将索引保存为JSON文件到/data文件夹
    output_dir = './data'
    os.makedirs(output_dir, exist_ok=True)  # exist_ok=True: 这个参数表示如果目录已经存在，不会抛出异常

    # 将DataFrame转换为JSON可序列化的格式
    train_dict = train_df.to_dict(orient='records')
    valid_dict = valid_df.to_dict(orient='records')

    # 保存训练集
    with open(os.path.join(output_dir, 'train_reviews.json'), 'w', encoding="utf8") as f:
        for record in train_dict:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    # 保存验证集
    with open(os.path.join(output_dir, 'valid_reviews.json'), 'w', encoding="utf8") as f:
        for record in valid_dict:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    print("数据集已划分并保存为JSON文件到/data文件夹下")
    print("训练集大小：", len(train_dict))
    print("验证集大小：", len(valid_dict))
    max_len, max_len_seq = get_max_len(train_dict, valid_dict)
    print("训练集和验证集的最大长度：", max_len)
    print("训练集和验证集的最大长度序列：", max_len_seq)
    print("训练集和验证集的平均长度：", get_avg_len(train_dict, valid_dict))

    # 获取并显示长度分布
    train_lengths = get_length_distribution(train_dict, "训练集")
    valid_lengths = get_length_distribution(valid_dict, "验证集")

    # 获取并显示正负样本统计信息
    train_label_dist = get_detailed_label_stats(train_dict, "训练集")
    valid_label_dist = get_detailed_label_stats(valid_dict, "验证集")
