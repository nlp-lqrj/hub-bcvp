# 20260111-week12-第十二周作业

## 作业内容和要求：梳理记录主流开源大模型的模型结构

## 1. 模型概述

主要分析了当前主流的十个大语言模型：BaiChuan、ChatGLM、DBRX、MOSS、DeepSeek、Gemma、Grok1、Llama 2、Mixtral和Qwen3，从位置编码、架构类型、Transformer结构、多头机制、FF层设计、归一化层选择、激活函数和bias使用等维度进行了详细对比。

## 2. 综合指标对比表

| 模型名称 | 架构类型 | 层数 | 隐藏层维度 | 最大序列长度 | 词汇表大小 | 位置编码类型 | 注意力头数 | 注意力特殊设计 | FF层类型 | 归一化层 | 激活函数 | Bias使用情况 |
|---------|---------|------|-----------|------------|-----------|-------------|-----------|--------------|---------|---------|---------|------------|
| BaiChuan | Decoder-only | 32 | 4096 | 4096 | 125696 | Rotary Embedding | 32 | 标准多头 | 带gate的MLP | RMSNorm | silu | 不使用 |
| ChatGLM | Decoder-only | 28 | 4096 | 32768 | 65024 | Rotary Embedding | 32 | 多查询注意力 | SwigLU MLP | RMSNorm | SwigLU | QKV层使用 |
| DBRX | Decoder-only | 40 | 6144 | 32768 | 100352 | Rotary Embedding | 48 | 分组查询注意力(8KV头) | MoE(16选4) | LayerNorm | GLU | 不使用 |
| MOSS | Decoder-only | 28 | 4096 | 2048 | 107008 | Rotary Embedding | 16 | 部分维度RoPE | 标准MLP | LayerNorm | gelu_new | LayerNorm使用 |
| DeepSeek | Decoder-only | 32 | 4096 | 4096 | 128000 | Rotary Embedding | 32 | 标准多头 | 带gate的MLP | RMSNorm | silu | 不使用 |
| Gemma 7B | Decoder-only | 28 | 3072 | 8192 | 256000 | Rotary Embedding | 16 | 标准多头 | 带gate的MLP | RMSNorm | silu | 不使用 |
| Grok1 | Decoder-only | 64 | 8192 | 8192 | 131072 | Rotary Embedding | 64 | 标准多头 | 带gate的MLP | RMSNorm | silu | 不使用 |
| Llama 2 7B | Decoder-only | 32 | 4096 | 4096 | 32000 | Rotary Embedding | 32 | 标准多头 | 带gate的MLP | RMSNorm | silu | 不使用 |
| Mixtral 8x7B | Decoder-only | 32 | 4096 | 32768 | 32000 | Rotary Embedding | 32 | 分组查询注意力(8KV头) | MoE(8选2) | RMSNorm | silu | 不使用 |
| Qwen3 | Decoder-only | 40 | 5120 | 128000 | 151936 | Rotary Embedding | 40 | 分组查询注意力 | 带gate的MLP | RMSNorm | SwigLU | 部分使用 |

## 3. 模型核心特点和使用场景

| 模型名称 | 核心特点                                                                      | 适用场景 |
|---------|---------------------------------------------------------------------------|---------|
| BaiChuan | 标准Transformer Decoder架构<br>不使用bias，减少参数量<br>使用silu激活函数<br>适中的模型规模         | 通用语言生成任务<br>对话系统<br>文本摘要<br>代码生成 |
| ChatGLM | 支持32768长序列<br>采用多查询注意力，提高推理速度<br>使用SwigLU激活函数<br>灵活的bias配置                | 长文档理解与生成<br>对话系统<br>知识密集型任务<br>多轮对话 |
| DBRX | 采用MoE架构（16选4）<br>大隐藏层维度（6144）<br>大theta值RoPE，适合长序列<br>分组查询注意力             | 大规模语言生成<br>复杂推理任务<br>长文档处理<br>高资源场景下的高性能模型 |
| MOSS | 部分维度应用RoPE，减少计算量<br>使用gelu_new激活函数<br>适中的模型规模<br>支持插件扩展                   | 通用语言生成<br>对话系统<br>插件增强的AI助手<br>中等资源部署 |
| DeepSeek | 与Llama 2架构一致，兼容性好<br>不使用bias，减少参数量<br>使用silu激活函数<br>适中的模型规模               | 通用语言生成<br>对话系统<br>代码生成<br>中等资源部署 |
| Gemma 7B | 谷歌开发的轻量级模型<br>隐藏层维度3072，适合资源受限场景<br>大词汇表（256000），支持多语言<br>使用RMSNorm和silu  | 边缘设备部署<br>轻量级对话系统<br>多语言任务<br>资源受限环境 |
| Grok1 | xAI开发的大型模型<br>深层数（64层）和大隐藏层维度（8192）<br>大注意力头数（64）<br>支持8192长序列            | 复杂推理任务<br>大规模语言生成<br>长文档处理<br>高资源场景 |
| Llama 2 7B | Meta开发的开源模型，生态丰富<br>标准Transformer Decoder架构<br>不使用bias，减少参数量<br>广泛的社区支持和工具链 | 通用语言生成<br>微调与定制化开发<br>研究实验<br>中等资源部署 |
| Mixtral 8x7B | Mistral AI开发的MoE模型（8选2）<br>支持32768长序列<br>分组查询注意力，提高推理速度<br>平衡性能与效率        | 高效语言生成<br>长文档处理<br>资源受限的高性能需求<br>大规模部署 |
| Qwen3 | 阿里巴巴开发的大模型<br>支持128000超长序列<br>分组查询注意力，提高推理效率<br>使用SwigLU激活函数              | 超长文档理解与生成<br>知识密集型任务<br>多轮对话<br>企业级应用 |


## 4. 架构对比总结

### 4.1 位置编码趋势
- **统一趋势**：所有模型均采用Rotary Embedding，这是当前大语言模型的主流选择
- **长序列优化**：DBRX使用了大theta值（500000），适合处理更长的序列
- **折中方案**：MOSS仅在部分维度应用RoPE，减少计算量的同时保留位置信息
- **序列长度多样化**：从2048（MOSS）到128000（Qwen3），模型支持的序列长度差异显著

### 4.2 多头注意力发展
- **标准多头注意力仍是基础**：BaiChuan、DeepSeek、Gemma、Grok1、Llama 2等模型均采用标准多头注意力
- **高效注意力机制兴起**：
  - 多查询注意力（ChatGLM）：提高推理速度
  - 分组查询注意力（DBRX、Mixtral、Qwen3）：平衡性能和效果
- **头数设计多样化**：从16（MOSS、Gemma）到64（Grok1），头数越多，模型捕获不同位置关系的能力越强

### 4.3 FF层设计
- **演进路径**：从标准MLP（MOSS）→ 带gate的MLP（BaiChuan、DeepSeek、Gemma、Grok1、Llama 2、Qwen3）→ SwigLU（ChatGLM）→ MoE（DBRX、Mixtral）
- **MoE架构成为重要方向**：
  - DBRX：16个专家，每次选择4个
  - Mixtral：8个专家，每次选择2个
  - 能够显著提高模型容量同时控制计算成本
- **高效激活函数**：silu和SwigLU成为主流，gelu_new逐渐减少

### 4.4 归一化层选择
- **RMSNorm成为主流**：除DBRX和MOSS外，其他模型均使用RMSNorm，计算更高效
- **LayerNorm仍有应用**：DBRX和MOSS继续使用传统的LayerNorm
- **统一趋势**：所有模型均采用前层归一化（Pre-LN）设计

### 4.5 Bias使用策略
- **减少或不使用bias成为趋势**：BaiChuan、DeepSeek、DBRX、Gemma、Grok1、Llama 2、Mixtral等模型均不使用bias，以减少参数量和计算量
- **选择性使用bias**：
  - ChatGLM：仅在QKV层使用bias
  - Qwen3：部分使用bias
  - MOSS：仅在LayerNorm中使用bias

### 4.6 模型规模与资源效率
- **多样化规模**：从Gemma 7B（轻量级）到Grok1（大型模型），满足不同资源需求
- **资源效率优化**：通过MoE架构、高效注意力机制、减少bias等方式，提高模型性能与计算资源的比值
- **超长序列支持**：ChatGLM、DBRX、Mixtral、Qwen3等模型支持32768及以上序列长度，适合长文档处理

### 4.7 架构一致性与生态
- **Llama 2架构影响力大**：DeepSeek等模型采用与Llama 2一致的架构，受益于其丰富的生态和工具链
- **开源模型生态丰富**：Llama 2、Mixtral等开源模型拥有广泛的社区支持和工具链
- **企业模型差异化**：ChatGLM、Qwen3等企业模型在架构上进行了差异化设计，如超长序列支持、特殊注意力机制等

## 5. 结论与建议

### 5.1 模型选择建议
- **资源受限场景**：
  - 边缘设备：选择Gemma 7B，模型规模小，适合部署
  - 中等资源：选择BaiChuan、DeepSeek或Llama 2 7B，模型规模适中，生态良好
- **长序列处理**：
  - 32768序列：选择ChatGLM、DBRX或Mixtral 8x7B
  - 超长序列（128000）：选择Qwen3，支持业界最长序列之一
- **高性能要求**：
  - 极致性能：选择Grok1，深层数和大隐藏层维度
  - 平衡性能与效率：选择Mixtral 8x7B或DBRX，采用MoE架构
- **推理速度优先**：
  - 选择ChatGLM，支持多查询注意力
  - 选择Mixtral 8x7B，分组查询注意力+MoE架构
- **开源生态优先**：
  - 选择Llama 2 7B或Mixtral 8x7B，拥有广泛的社区支持和工具链
- **企业级应用**：
  - 选择Qwen3，支持超长序列和分组查询注意力

### 5.2 架构设计趋势
- **位置编码**：Rotary Embedding继续主导，针对长序列的优化（如大theta值）将成为重点
- **注意力机制**：高效注意力机制（多查询、分组查询）将继续发展，成为主流
- **FF层设计**：
  - MoE架构将被更广泛采用，专家数量和选择策略将进一步优化
  - 带gate的MLP成为标准设计
  - SwigLU等高效激活函数将替代传统激活函数
- **归一化层**：RMSNorm将完全取代LayerNorm，成为唯一选择
- **Bias使用**：减少或不使用bias成为普遍趋势
- **序列长度**：更长的序列支持（64K、128K甚至更长）将成为竞争焦点

### 5.3 未来发展方向
- **更长序列支持**：突破当前128K的限制，支持更长的文档处理
- **更高效的注意力机制**：探索新的注意力计算方法，进一步提高推理速度
- **更智能的MoE路由策略**：优化专家选择机制，提高MoE模型的效率和效果
- **更好的激活函数设计**：探索更高效、更强大的激活函数
- **更高效的训练和推理方法**：
  - 模型量化技术的进一步发展
  - 分布式训练优化
  - 推理加速技术
- **架构创新**：突破现有Transformer架构，探索新的模型结构
- **多模态融合**：更好地融合语言、图像、音频等多种模态
- **轻量级模型优化**：在保持性能的同时，进一步减小模型体积，适合边缘设备部署

## 6. 参考文献

1. BaiChuan模型源码 - [https://github.com/baichuan-inc/Baichuan2](https://github.com/baichuan-inc/Baichuan2)
2. ChatGLM模型源码 - [https://github.com/THUDM/ChatGLM3](https://github.com/THUDM/ChatGLM3)
3. DBRX模型源码 - [https://github.com/databricks/dbrx](https://github.com/databricks/dbrx)
4. MOSS模型源码 - [https://github.com/OpenLMLab/MOSS](https://github.com/OpenLMLab/MOSS)
5. DeepSeek模型相关资料 - [https://github.com/deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
6. Gemma模型相关资料 - [https://ai.google.dev/gemma](https://ai.google.dev/gemma)
7. Grok1模型相关资料 - [https://x.ai/products/grok](https://x.ai/products/grok)
8. Llama 2模型相关资料 - [https://github.com/facebookresearch/llama](https://github.com/facebookresearch/llama)
9. Mixtral模型相关资料 - [https://github.com/mistralai/mistral-src](https://github.com/mistralai/mistral-src)
10. Qwen3模型相关资料 - [https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen)
11. Attention Is All You Need - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
12. RoFormer: Enhanced Transformer with Rotary Position Embedding - [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
13. Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity - [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)
14. GLU Variants Improve Transformer - [https://arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202)
15. Rotary Position Embedding: A Relative Position Embedding for Transformers - [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)