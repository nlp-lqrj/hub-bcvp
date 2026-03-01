## 20260118-week13-第十三周作业

### 1. 作业内容和要求

#### 基于已有的项目，通过BERT和LoRA实现NER

- 本项目将传统基于LSTM的命名实体识别（NER）模型改造为基于BERT的实现，并集成LoRA（Low-Rank Adaptation）微调技术，以提升模型性能的同时减少训练参数量。
- BERT作为预训练语言模型，能够提供更丰富的上下文语义信息，显著改善序列标注任务的效果。
- LoRA技术通过低秩分解减少可训练参数，加快微调速度，降低显存需求。

### 2. 改造内容

#### 2.1 配置文件（ner/config.py）

**主要修改**：
- 配置`use_bert`参数控制是否使用BERT模型
- 指定`bert_path`指向本地预训练BERT模型

```python
Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,  # 暂时禁用CRF以避免依赖问题
    "use_bert": True,  # 控制是否使用BERT
    "class_num": 9,
    "bert_path": "E:\\BaiduNetdiskDownload\\bert-base-chinese",  # BERT模型路径
    "tuning_tactics": "lora_tuning",  # 微调策略
    "seed": 987  # 随机种子
}
```

#### 2.2 数据加载模块（ner/loader.py）

**核心修改**：
- 引入BERT分词器替代自定义字表
- 实现子词标签对齐逻辑
- 处理BERT特殊token和padding

```python
# 导入BERT快速分词器
from transformers import BertTokenizerFast

# 初始化BERT分词器
self.use_bert = "bert_path" in config and config["bert_path"]  # 字典键检查替代hasattr
if self.use_bert:
    self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])
else:
    self.vocab = load_vocab(config["vocab_path"])
    self.config["vocab_size"] = len(self.vocab)

# 子词标签对齐逻辑
encoded = self.tokenizer(text, padding=False, truncation=False, add_special_tokens=False, return_offsets_mapping=True)
input_ids = encoded["input_ids"]
offset_mapping = encoded["offset_mapping"]

# 扩展标签以匹配子词序列长度
extended_labels = []
char_idx = 0
for token_offset in offset_mapping:
    if token_offset == (0, 0):
        extended_labels.append(-1)
    else:
        extended_labels.append(labels[char_idx])
        if token_offset[1] > (char_idx + 1):
            char_idx += 1
```

#### 2.3 模型架构（ner/model.py）

**主要修改**：
- 引入BertModel作为特征提取器
- 集成LoRA微调技术
- 添加参数分组优化策略
- 保留原有LSTM结构作为备选

```python
# 导入BERT模型和LoRA相关库
from transformers import BertModel, AutoModelForTokenClassification
from peft import get_peft_model, LoraConfig

# 直接使用AutoModelForTokenClassification作为基础模型
TorchModel = AutoModelForTokenClassification.from_pretrained(
    Config["bert_path"],
    num_labels=Config["class_num"]
)

# 配置LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"]
)

# 应用LoRA到模型
model = get_peft_model(TorchModel, peft_config)

# 确保分类器层可训练
for param in model.get_submodule("model").get_submodule("classifier").parameters():
    param.requires_grad = True

# 参数分组优化
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    
    if hasattr(model, "use_bert") and model.use_bert:
        # 对BERT模型进行参数分组
        bert_params = list(model.bert.named_parameters())
        other_params = list(model.classify.named_parameters())
        
        param_groups = [
            {'params': [p for n, p in bert_params], 'lr': learning_rate * 0.1},  # BERT参数学习率×0.1
            {'params': [p for n, p in other_params], 'lr': learning_rate}  # 其他参数使用正常学习率
        ]
        
        if optimizer == "adam":
            return Adam(param_groups)
        elif optimizer == "sgd":
            return SGD(param_groups)
    else:
        # 原有优化器设置
        if optimizer == "adam":
            return Adam(model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            return SGD(model.parameters(), lr=learning_rate)
```

#### 2.4 依赖安装

```bash
pip install transformers peft
```

### 3. 执行过程

#### 3.1 初始化与配置
1. 确认本地BERT模型路径有效性
2. 配置文件中设置`use_bert=True`
3. 安装必要依赖

#### 3.2 数据预处理
1. BERT分词器加载与初始化
2. 子词分割与标签对齐
3. 输入序列padding与truncation

#### 3.3 模型训练
1. BERT模型参数加载
2. 配置并应用LoRA
3. 参数分组优化器初始化
4. 训练循环执行
5. 保存LoRA权重

#### 3.4 关键问题解决

| 问题类型 | 错误信息 | 解决方案 |
|---------|---------|---------|
| 依赖缺失 | `ModuleNotFoundError: No module named 'transformers'` | 安装transformers库 |
| 依赖缺失 | `ModuleNotFoundError: No module named 'peft'` | 安装peft库 |
| 路径错误 | `OSError: Incorrect path_or_model_id` | 检查并修正BERT模型路径 |
| 属性错误 | `hasattr(config, "bert_path")` | 改为字典键检查：`"bert_path" in config` |
| 版本兼容 | `AttributeError: 'tuple' object has no attribute 'last_hidden_state'` | 使用索引访问：`outputs[0]` |
| 功能缺失 | `return_offset_mapping is not available when using Python tokenizers` | 改用`BertTokenizerFast` |
| CUDA错误 | `Assertion 't >= 0 && t < n_classes' failed` | 确保标签值在有效范围内，使用`ignore_index=-1` |
| 长度错误 | `AssertionError: assert len(labels) == len(pred_results) == len(sentences)` | 确保评估时长度一致 |

### 4. 结果对比

#### 4.1 训练损失

| Epoch | 平均损失（BERT + LoRA） | 平均损失（BERT） | 平均损失（LSTM） |
|-------|------------------------|-----------------|-----------------|
| 1     | 0.601021               | 21.645580       | ~35.0           |
| 2     | 0.297355               | 9.083621        | ~22.0           |
| 3     | 持续降低               | 持续降低        | ~15.0           |

#### 4.2 验证性能

**BERT + LoRA模型第2轮验证结果**：
```
PERSON类实体，准确率：0.493506, 召回率: 0.393782, F1: 0.438035
LOCATION类实体，准确率：0.526882, 召回率: 0.206751, F1: 0.296966
TIME类实体，准确率：0.279570, 召回率: 0.149425, F1: 0.194752
ORGANIZATION类实体，准确率：0.491525, 召回率: 0.305263, F1: 0.376619
Macro-F1: 0.326593
Micro-F1: 0.327864
```

**BERT模型第2轮验证结果**：
```
PERSON类实体，准确率：0.685714, 召回率: 0.746114, F1: 0.714635
LOCATION类实体，准确率：0.630137, 召回率: 0.582278, F1: 0.605258
TIME类实体，准确率：0.450617, 召回率: 0.419540, F1: 0.434519
ORGANIZATION类实体，准确率：0.570175, 召回率: 0.684210, F1: 0.622005
Macro-F1: 0.594104
Micro-F1: 0.598286
```

**原有LSTM模型参考结果**：
```
Macro-F1: ~0.45
Micro-F1: ~0.48
```

#### 4.3 性能提升对比

| 指标 | BERT + LoRA模型 | BERT模型 | LSTM模型 | LoRA相比LSTM提升 |
|------|----------------|---------|---------|------------------|
| Macro-F1 | 0.3266 | 0.5941 | ~0.45 | -27.4%（初期结果） |
| Micro-F1 | 0.3279 | 0.5983 | ~0.48 | -31.7%（初期结果） |
| PERSON F1 | 0.4380 | 0.7146 | ~0.62 | -29.5%（初期结果） |
| LOCATION F1 | 0.2970 | 0.6053 | ~0.50 | -40.6%（初期结果） |
| ORGANIZATION F1 | 0.3766 | 0.6220 | ~0.48 | -21.5%（初期结果） |
| TIME F1 | 0.1948 | 0.4345 | ~0.32 | -39.2%（初期结果） |

**注**：上述结果为LoRA模型训练初期的表现，随着训练轮次的增加，性能会进一步提升。LoRA的主要优势在于减少可训练参数和显存使用，而不是立即获得最佳性能。

### 5. 关键技术点

#### 5.1 子词对齐机制
BERT采用子词分词算法，需要将字符级别的标签扩展到子词序列。通过`offset_mapping`实现标签与子词的准确对齐，确保每个子词都获得正确的实体标签。

#### 5.2 参数分组优化
对BERT预训练参数使用较小的学习率（lr×0.1），对分类层和CRF层使用正常学习率，避免预训练参数被过度调整，同时让新添加的层能够快速适应任务。

#### 5.3 双模型兼容设计
保留原有LSTM结构作为备选，通过`use_bert`配置参数实现模型切换，提高代码的灵活性和可维护性。

#### 5.4 数据处理适配
针对BERT输入格式调整数据预处理流程，包括：
- 添加特殊标记（[CLS]、[SEP]）
- 子词分割与标签扩展
- 输入序列padding与truncation

#### 5.5 LoRA微调技术
- **低秩分解**：通过矩阵分解减少可训练参数，使用r=8的低秩矩阵
- **参数高效**：仅训练LoRA添加的参数，冻结预训练模型权重
- **目标模块**：针对注意力机制的query、key、value矩阵应用LoRA
- **灵活性**：支持在不同层应用不同的LoRA配置

### 6. 结论

- 本次BERT改造显著提升了NER模型的性能，各实体类型的F1值均有明显提升。
- 集成LoRA微调技术后，虽然初期性能略低于标准BERT模型，但显著减少了可训练参数，降低了显存需求。
- BERT作为预训练语言模型，能够捕捉更丰富的上下文语义信息，为序列标注任务提供更优质的特征表示。
- LoRA技术通过低秩分解实现参数高效微调，为大模型在资源受限设备上的部署提供了可能。

#### 优势
1. **性能提升**：标准BERT模型Macro-F1提升32%，Micro-F1提升24.6%
2. **参数高效**：LoRA仅需训练少量参数，减少显存使用
3. **语义理解**：BERT的双向编码器结构提供更全面的上下文信息
4. **灵活性**：支持BERT和LSTM模型的灵活切换，以及多种微调策略
5. **扩展性**：为后续模型优化和迁移学习提供基础

#### 挑战
1. **初期性能**：LoRA模型初期性能可能低于标准微调
2. **调参复杂度**：需要合理选择LoRA的秩r和其他超参数
3. **兼容性**：需要确保与不同版本的transformers库兼容

#### 后续改进方向
1. 尝试不同的预训练BERT模型
2. 优化子词对齐策略
3. 调整LoRA的超参数（r值、学习率等）
4. 探索半监督学习方法
5. 结合其他参数高效微调技术（如Prefix Tuning、Prompt Tuning）
6. 测试不同的目标模块组合

### 7. 环境配置

- Python版本：3.12
- 主要依赖：
  - torch
  - transformers
  - peft  # LoRA所需
  - numpy
  - sklearn

**安装命令**：
```bash
pip install torch transformers peft numpy sklearn
```

