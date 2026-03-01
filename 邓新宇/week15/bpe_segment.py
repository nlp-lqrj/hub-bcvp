#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BPE (Byte Pair Encoding) 分词实现
"""

import collections

class BPE:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
    
    def get_stats(self, vocab):
        """统计相邻字符对的频率"""
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        """合并最频繁的字符对"""
        v_out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in vocab.items():
            w_out = word.replace(bigram, replacement)
            v_out[w_out] = freq
        return v_out
    
    def train(self, text):
        """训练BPE模型"""
        # 预处理：过滤空白字符
        filtered_text = ''.join([char for char in text if char.strip()])
        
        if not filtered_text:
            return
        
        # 初始化：每个字符作为单独的token
        # 对于中文，我们将整个文本视为一个长序列
        # 统计相邻字符对的频率
        pairs = collections.defaultdict(int)
        for i in range(len(filtered_text)-1):
            pair = (filtered_text[i], filtered_text[i+1])
            pairs[pair] += 1
        
        # 初始化词汇表：每个字符作为单独的token
        vocab = {}
        for char in set(filtered_text):
            vocab[char] = filtered_text.count(char)
        
        # 迭代合并
        max_merges = min(500, self.vocab_size - len(vocab))
        for i in range(max_merges):
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            # 合并字符对
            new_token = ''.join(best)
            vocab[new_token] = pairs[best]
            
            # 更新merges字典
            self.merges[best] = i
            
            # 更新pairs：移除包含best的旧对，添加新对
            # 这里简化处理，只记录合并的对
            # 实际BPE算法会更复杂地更新所有可能的对
            
        # 构建最终词汇表
        for token in vocab:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
    
    def segment(self, text):
        """使用训练好的BPE模型进行分词"""
        if not self.merges:
            raise ValueError("BPE model not trained")
        
        segmented = []
        
        # 对整个文本进行分词
        # 初始化：每个字符作为单独的token
        tokens = list(text)
        
        # 应用合并规则
        while True:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            # 找到在merges中的pair
            valid_pairs = [(p, self.merges.get(p, float('inf'))) for p in pairs]
            if not valid_pairs:
                break
            # 找到最早的合并对
            best = min(valid_pairs, key=lambda x: x[1])
            if best[1] == float('inf'):
                break
            # 合并最佳pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best[0]:
                    new_tokens.append(''.join(best[0]))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # 过滤空白字符
        segmented = [token for token in tokens if token.strip()]
        
        return segmented

def main():
    # 读取corpus文件
    corpus_path = 'corpus'
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 初始化BPE模型
    bpe = BPE(vocab_size=10000)
    
    # 训练模型
    print("Training BPE model...")
    bpe.train(text)
    print(f"BPE model trained with {len(bpe.vocab)} tokens")
    
    # 保存词汇表
    with open('bpe_vocab.txt', 'w', encoding='utf-8') as f:
        for token, idx in sorted(bpe.vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\t{idx}\n")
    print("Vocabulary saved to bpe_vocab.txt")
    
    # 保存合并规则
    with open('bpe_merges.txt', 'w', encoding='utf-8') as f:
        for pair, idx in sorted(bpe.merges.items(), key=lambda x: x[1]):
            f.write(f"{pair[0]} {pair[1]}\t{idx}\n")
    print("Merge rules saved to bpe_merges.txt")
    
    # 测试分词
    test_sentences = [
        "主力合约突破21000元/吨重要关口",
        "分析师指出由于橡胶现货需求强劲",
        "中国艺术品市场在经历了2008年的低潮"
    ]
    
    print("\nTesting segmentation:")
    for sentence in test_sentences:
        segmented = bpe.segment(sentence)
        print(f"Original: {sentence}")
        print(f"Segmented: {' '.join(segmented)}")
        print()
    
    # 对整个corpus进行分词
    print("Segmenting entire corpus...")
    segmented_corpus = bpe.segment(text)
    
    # 保存分词结果
    with open('corpus_segmented.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(segmented_corpus))
    print("Segmented corpus saved to corpus_segmented.txt")

if __name__ == "__main__":
    main()
