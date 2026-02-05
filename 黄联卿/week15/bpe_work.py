# encoding: utf-8
import os


# ==================== BPE核心函数 ====================
def get_stats(ids):
    """统计字节对频率"""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """合并字节对"""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# ==================== 编码函数 ====================
def encode_text(text, vocab_size=276, verbose=True):
    """
    对文本进行BPE编码
    verbose: 是否打印合并过程的详细信息
    """
    # 初始化为UTF-8字节
    tokens = text.encode("utf-8")
    ids = list(map(int, tokens))

    # 训练BPE合并规则
    num_merges = vocab_size - 256
    merges = {}

    print(f"开始BPE训练，目标词汇表大小: {vocab_size}")
    print(f"初始字节数: {len(ids)}")
    print(f"计划合并次数: {num_merges}")
    print("-" * 60)

    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:  # 如果没有更多可合并的字节对
            if verbose:
                print(f"\n提前停止：没有更多可合并的字节对（已完成 {i} 次合并）")
            break

        # 找出最高频的字节对
        pair = max(stats, key=stats.get)
        idx = 256 + i

        # 显示合并信息
        if verbose:
            # 将字节对转换为可读形式
            def byte_to_readable(b):
                if b == 32:
                    return "' ' (space)"
                elif 32 <= b <= 126:  # 可打印ASCII字符
                    return f"'{chr(b)}'"
                else:
                    return f"\\x{b:02x}"

            pair_str = f"({pair[0]}, {pair[1]})"
            freq = stats[pair]

            print(f"merging {pair_str} into a new token {idx} 出现次数: {freq}")

        # 执行合并
        ids = merge(ids, pair, idx)
        merges[pair] = idx

        # 每10次合并显示一次进度
        if verbose and (i + 1) % 10 == 0:
            print(f"  [进度: {i + 1}/{num_merges}] 当前序列长度: {len(ids)}")

    if verbose:
        print("-" * 60)
        print(f"BPE训练完成")
        print(f"最终序列长度: {len(ids)}")
        print(f"实际合并次数: {len(merges)}")
        print(f"压缩率: {len(tokens) / len(ids):.2f}X")

    return ids, merges


def create_vocab(merges, verbose=True):
    """
    根据合并规则创建词汇表
    """
    # 初始化基本字节词汇表
    vocab = {idx: bytes([idx]) for idx in range(256)}

    # 添加合并的词汇
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    # 显示部分词汇表
    if verbose:
        print("\n=== BPE词汇表摘要 ===")
        print("基本字节 (0-255): 已包含所有单字节")
        print("合并词汇:")

        # 显示前20个合并词汇
        vocab_items = [(idx, vocab[idx]) for idx in range(256, 256 + len(merges))]
        for idx, byte_val in vocab_items[:20]:
            try:
                decoded = byte_val.decode('utf-8', errors='replace')
                print(f"ID {idx}: {decoded!r} (来自: {list(byte_val)})")
            except:
                print(f"ID {idx}: 无法解码 (原始字节: {list(byte_val)})")

    return vocab


# ==================== 解码函数 ====================
def decode_ids(ids, vocab):
    """
    解码ID列表为文本
    """
    # 使用词汇表解码字节
    tokens = b"".join(vocab[idx] for idx in ids)
    # 解码为UTF-8文本
    text = tokens.decode("utf-8", errors="replace")
    return text


# ==================== 独立入口函数 - 编码（含BPE训练） ====================
def encode_main(input_file="corpus.txt", vocab_size=276):
    """
    【编码专用入口】处理文件：训练BPE、编码文本、保存编码结果和词汇表
    生成文件：corpus_encoded.txt / bpe_vocabulary.txt
    """
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：文件 '{input_file}' 不存在！")
        print("请确保corpus.txt文件在当前目录中")
        return None, None

    print("=" * 60)
    print(f"BPE编码/训练系统 - 处理文件: {input_file}")
    print("=" * 60)

    # 1. 读取原始文本
    with open(input_file, 'r', encoding='utf-8') as f:
        original_text = f.read()

    print(f"原始文本长度: {len(original_text)} 字符")
    print(f"原始字节数: {len(original_text.encode('utf-8'))}")

    # 2. 编码文本（这里会显示合并过程）
    print("\n" + "=" * 60)
    print("开始BPE训练过程...")
    print("=" * 60)

    encoded_ids, merges = encode_text(original_text, vocab_size, verbose=True)

    print(f"\n编码完成！编码后的ID数量: {len(encoded_ids)}")

    # 3. 显示部分编码结果
    print("\n=== 部分编码结果（前50个ID） ===")
    for i in range(0, min(50, len(encoded_ids)), 10):
        chunk = encoded_ids[i:i + 10]
        print(f"ID {i:3}-{i + len(chunk) - 1:3}: {chunk}")
    print(f"... 总共 {len(encoded_ids)} 个ID")

    # 4. 保存编码结果
    encoded_file = "corpus_encoded.txt"
    with open(encoded_file, 'w', encoding='utf-8') as f:
        # 保存为逗号分隔的ID列表
        f.write(','.join(map(str, encoded_ids)))

    print(f"\n编码结果已保存到: {encoded_file}")

    # 5. 创建词汇表
    vocab = create_vocab(merges, verbose=True)

    # 6. 保存完整词汇表
    vocab_file = "bpe_vocabulary.txt"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write("BPE词汇表\n")
        f.write("=" * 40 + "\n")
        f.write(f"词汇表大小: {len(vocab)}\n")
        f.write(f"合并规则数量: {len(merges)}\n\n")

        # 保存所有合并规则
        f.write("合并规则 (按学习顺序):\n")
        for i, ((p0, p1), idx) in enumerate(merges.items()):
            f.write(f"{i:3}. merging ({p0:3}, {p1:3}) -> ID {idx:3}\n")

        f.write("\n" + "=" * 40 + "\n")
        f.write("词汇表内容:\n\n")

        # 保存基本字节
        f.write("基本字节 (0-255):\n")
        for idx in range(256):
            byte_val = vocab[idx]
            try:
                decoded = byte_val.decode('utf-8', errors='replace')
                if decoded.isprintable() and decoded != '\n':
                    f.write(f"ID {idx:3}: '{decoded}' (字节: {list(byte_val)})\n")
                else:
                    f.write(f"ID {idx:3}: 非打印字符 (字节: {list(byte_val)})\n")
            except:
                f.write(f"ID {idx:3}: 特殊字节 (字节值: {list(byte_val)})\n")

        # 保存合并的词汇
        f.write("\n合并词汇 (256+):\n")
        for idx in range(256, len(vocab)):
            if idx in vocab:
                byte_val = vocab[idx]
                try:
                    decoded = byte_val.decode('utf-8', errors='replace')
                    f.write(f"ID {idx:3}: '{decoded}' (来自: {list(byte_val)})\n")
                except:
                    f.write(f"ID {idx:3}: 组合字节 (原始: {list(byte_val)})\n")

    print(f"\n完整词汇表已保存到: {vocab_file}")
    print("=" * 60)
    print("编码/训练流程完成！")
    print("=" * 60)
    return encoded_ids, vocab


# ==================== 独立入口函数 - 解码 ====================
def decode_main(encoded_file="corpus_encoded.txt", vocab_file="bpe_vocabulary.txt"):
    """
    【解码专用入口】从文件读取编码ID和词汇表，解码为文本并验证
    依赖文件：corpus_encoded.txt / bpe_vocabulary.txt（由encode_main生成）
    生成文件：corpus_decoded.txt
    """
    # 检查依赖文件是否存在
    if not os.path.exists(encoded_file):
        print(f"错误：编码文件 '{encoded_file}' 不存在！")
        print("请先执行encode_main生成编码文件")
        return None
    if not os.path.exists(vocab_file):
        print(f"错误：词汇表文件 '{vocab_file}' 不存在！")
        print("请先执行encode_main生成词汇表文件")
        return None

    print("=" * 60)
    print(f"BPE解码系统 - 处理文件")
    print("=" * 60)

    # 1. 读取编码ID
    print("1. 读取编码ID列表...")
    with open(encoded_file, 'r', encoding='utf-8') as f:
        encoded_ids = list(map(int, f.read().split(',')))
    print(f"成功读取 {len(encoded_ids)} 个编码ID")

    # 2. 解析词汇表文件重建vocab
    print("\n2. 解析词汇表文件重建词汇表...")
    vocab = {idx: bytes([idx]) for idx in range(256)}  # 初始化基础字节
    with open(vocab_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        merge_start = False
        for line in lines:
            line = line.strip()
            # 匹配合并规则行：xxx. merging ( xxx,  xxx) -> ID  xxx
            if line.startswith(tuple(str(i) for i in range(10))) and "merging" in line and "->" in line:
                merge_start = True
                # 提取合并对和目标ID
                try:
                    # 截取 (p0,p1) 部分
                    pair_part = line.split("merging")[-1].split("->")[0].strip()
                    p0 = int(pair_part.strip('()').split(',')[0].strip())
                    p1 = int(pair_part.strip('()').split(',')[1].strip())
                    # 提取目标ID
                    idx = int(line.split("ID")[-1].strip())
                    vocab[idx] = vocab[p0] + vocab[p1]
                except:
                    continue
    print(f"成功重建词汇表，总大小: {len(vocab)} (基础256 + 合并{len(vocab)-256})")

    # 3. 执行解码
    print("\n3. 开始解码ID为文本...")
    decoded_text = decode_ids(encoded_ids, vocab)
    print("解码完成！")

    # 4. 显示部分解码结果
    print("\n=== 部分解码结果（前200字符） ===")
    print(decoded_text[:200])
    if len(decoded_text) > 200:
        print("...")

    # 5. 保存解码结果
    decoded_file = "corpus_decoded.txt"
    with open(decoded_file, 'w', encoding='utf-8') as f:
        f.write(decoded_text)
    print(f"\n解码结果已保存到: {decoded_file}")

    # 6. 验证（如果原始文件存在则对比）
    print("\n4. 编解码一致性验证...")
    original_file = "corpus.txt"
    if os.path.exists(original_file):
        with open(original_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
        if original_text == decoded_text:
            print("✓ 编解码完全正确！原始文本和解码文本完全一致。")
        else:
            # 检查差异
            if len(original_text) != len(decoded_text):
                print(f"⚠ 长度不一致: 原始={len(original_text)}, 解码={len(decoded_text)}")
            else:
                diff_count = 0
                for i, (orig, dec) in enumerate(zip(original_text, decoded_text)):
                    if orig != dec:
                        if diff_count < 5:  # 只显示前5个差异
                            print(f"⚠ 第{i}个字符不同: 原始={orig!r}, 解码={dec!r}")
                        diff_count += 1
                if diff_count == 0:
                    print("✓ 文本内容一致")
                else:
                    print(f"⚠ 发现 {diff_count} 处差异")
    else:
        print(f"提示：原始文件 {original_file} 不存在，跳过一致性验证")

    # 7. 压缩统计
    if os.path.exists(original_file):
        original_bytes = len(original_text.encode('utf-8'))
        encoded_count = len(encoded_ids)
        compression_ratio = original_bytes / encoded_count if encoded_count > 0 else 0
        print(f"\n=== 压缩统计 ===")
        print(f"原始字节数: {original_bytes}")
        print(f"编码后ID数量: {encoded_count}")
        print(f"压缩率: {compression_ratio:.2f}X")
        print(f"数据减少: {(1 - encoded_count / original_bytes) * 100:.1f}%")

    print("=" * 60)
    print("解码流程完成！")
    print("=" * 60)
    return decoded_text


# ==================== 主程序入口 - 可选择测试编码/解码 ====================
if __name__ == "__main__":
    # 配置参数
    VOCAB_SIZE = 276  # BPE目标词汇表大小

    # ------------------- 测试选择区 -------------------
    # 测试编码（含BPE训练）：取消下面一行注释，注释解码行
    encode_main(input_file="corpus.txt", vocab_size=VOCAB_SIZE)

    # 测试解码：取消下面一行注释，注释编码行（需先执行编码生成依赖文件）
    # decode_main(encoded_file="corpus_encoded.txt", vocab_file="bpe_vocabulary.txt")
