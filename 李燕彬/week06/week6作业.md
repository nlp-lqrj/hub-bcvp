## 20251123-week06-第六周作业

### 1 作业内容和要求

[week06作业.py](week06%E4%BD%9C%E4%B8%9A.py)

#### 用pytorch实现transformer层

#### 代码说明:

##### 1.多头注意力（MultiHeadAttention）

- 将 Q/K/V 拆分为多个头并行计算注意力，最后拼接输出，增强模型对不同特征的捕捉能力
- 通过缩放因子（√d_k）避免点积结果过大导致 softmax 饱和

##### 2. 位置前馈网络（PositionwiseFeedforward）
- 对每个位置的特征独立进行两次线性变换和 ReLU 激活，增强模型的非线性拟合能力

##### 3.Transformer 层（TransformerLayer）
- 遵循 “Self-Attention → 残差连接 + 层归一化 → 前馈网络 → 残差连接 + 层归一化” 的经典流程
- 层归一化（LayerNorm）在残差连接后应用（Post-LN 结构），是 Transformer 的标准实现方式

##### 4.掩码（Mask）
- 用于屏蔽 padding 部分或未来 token（Decoder 层），确保模型只关注有效信息

#### 使用场景
- 可直接堆叠多个TransformerLayer构建完整的 Transformer Encoder
- 若需实现 Decoder 层，需额外添加编码器 - 解码器注意力层（Encoder-Decoder Attention）


```bash
python3 week06作业.py
```
###### 执行脚本运行结果如下
```bash
输入形状: torch.Size([2, 5, 128])
输出形状: torch.Size([2, 5, 128])
注意力权重形状: torch.Size([2, 8, 5, 5])
```