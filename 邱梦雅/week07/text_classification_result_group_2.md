
### 预训练模型文本分类效果对比
说明：
1. 以下模型训练均使用bert自带的`vocab.txt`词表，词表大小为21128
2. 训练集和测试集的出入文本大小采用`max_length`为30或40，整体数据集的**平均文本长度为25**，可以完全覆盖60%左右的样本
3. `bert`、`bert_lstm`、`bert_cnn`和`bert_mid_layer`均采用`num_layers`6层transformer结构
4. `bert_lstm`模型使用默认1层`lstm`结构，`bert_cnn`模型使用`cnn`层的`kernel_size`为3
5. `bert_mid_layer`模型选择性地使用`bert`模型的**第2、4、6层**输出进行相加计算结果
6. 所有数据集训练轮数`epoch`皆为15轮
7. 对于`bert`类预训练模型，使用**低学习率learning_rate**会有更好的训练效果
8. `bert`类预训练模型只使用了`max_length`为**30或40**的训练样本，预测准确率与非预训练模型使用`max_length`为**75**的训练样本几乎持平
9. 从实验结果来看，整体上电商评论数据集文本分类效果从高到底排序：bert模型 > bert_mid_layer模型 > bert_lstm模型 > bert_cnn模型


| model_type     |   max_length |   hidden_size | pooling_style   |   batch_size |   learning_rate | optimizer   |   epoch |   accuracy |
|----------------|--------------|---------------|-----------------|--------------|-----------------|-------------|---------|------------|
| bert           |           40 |           768 | max             |          128 |          1e-05  | adam        |      15 |   0.907423 |
| bert           |           40 |           768 | max             |           64 |          1e-05  | adam        |      15 |   0.904921 |
| bert           |           30 |           768 | avg             |          128 |          0.0001 | adam        |      15 |   0.901585 |
| bert           |           40 |           768 | max             |           64 |          0.0001 | adam        |      15 |   0.901168 |
| bert_mid_layer |           40 |           768 | max             |           64 |          0.0001 | adam        |      15 |   0.900751 |
| bert           |           40 |           768 | max             |          128 |          0.0001 | adam        |      15 |   0.900334 |
| bert           |           30 |           768 | avg             |           64 |          0.0001 | adam        |      15 |   0.899917 |
| bert           |           30 |           768 | max             |          128 |          1e-05  | adam        |      15 |   0.899917 |
| bert           |           40 |           768 | avg             |           64 |          1e-05  | adam        |      15 |   0.8995   |
| bert           |           40 |           768 | avg             |          128 |          1e-05  | adam        |      15 |   0.8995   |
| bert_mid_layer |           40 |           768 | max             |          128 |          0.0001 | adam        |      15 |   0.898666 |
| bert           |           30 |           768 | avg             |          128 |          1e-05  | adam        |      15 |   0.898249 |
| bert           |           40 |           768 | avg             |           64 |          0.0001 | adam        |      15 |   0.89658  |
| bert_mid_layer |           40 |           768 | max             |           64 |          1e-05  | adam        |      15 |   0.896163 |
| bert_mid_layer |           30 |           768 | max             |           64 |          0.0001 | adam        |      15 |   0.894495 |
| bert_mid_layer |           40 |           768 | max             |          128 |          1e-05  | adam        |      15 |   0.894495 |
| bert           |           30 |           768 | max             |           64 |          0.0001 | adam        |      15 |   0.894078 |
| bert           |           30 |           768 | avg             |           64 |          1e-05  | adam        |      15 |   0.894078 |
| bert_mid_layer |           30 |           768 | max             |          128 |          0.0001 | adam        |      15 |   0.893661 |
| bert_mid_layer |           40 |           768 | avg             |          128 |          1e-05  | adam        |      15 |   0.892827 |
| bert_lstm      |           40 |           768 | avg             |           64 |          0.0001 | adam        |      15 |   0.891993 |
| bert           |           30 |           768 | max             |           64 |          1e-05  | adam        |      15 |   0.891576 |
| bert_lstm      |           40 |           768 | max             |          128 |          0.0001 | adam        |      15 |   0.890742 |
| bert_cnn       |           40 |           768 | max             |           64 |          1e-05  | adam        |      15 |   0.889908 |
| bert_mid_layer |           40 |           768 | avg             |           64 |          1e-05  | adam        |      15 |   0.889908 |
| bert_lstm      |           40 |           768 | max             |           64 |          0.0001 | adam        |      15 |   0.889491 |
| bert_mid_layer |           30 |           768 | max             |           64 |          1e-05  | adam        |      15 |   0.889491 |
| bert           |           30 |           768 | max             |          128 |          0.0001 | adam        |      15 |   0.889074 |
| bert_mid_layer |           40 |           768 | avg             |          128 |          0.0001 | adam        |      15 |   0.886572 |
| bert_lstm      |           30 |           768 | avg             |          128 |          0.0001 | adam        |      15 |   0.886155 |
| bert           |           40 |           768 | avg             |          128 |          0.0001 | adam        |      15 |   0.884904 |
| bert_mid_layer |           40 |           768 | avg             |           64 |          0.0001 | adam        |      15 |   0.88407  |
| bert_lstm      |           30 |           768 | avg             |           64 |          1e-05  | adam        |      15 |   0.883653 |
| bert_cnn       |           40 |           768 | avg             |           64 |          0.0001 | adam        |      15 |   0.883653 |
| bert_lstm      |           40 |           768 | avg             |          128 |          0.0001 | adam        |      15 |   0.882819 |
| bert_cnn       |           40 |           768 | avg             |          128 |          0.0001 | adam        |      15 |   0.882819 |
| bert_cnn       |           40 |           768 | max             |          128 |          0.0001 | adam        |      15 |   0.882402 |
| bert_lstm      |           40 |           768 | max             |          128 |          1e-05  | adam        |      15 |   0.881985 |
| bert_cnn       |           30 |           768 | avg             |           64 |          1e-05  | adam        |      15 |   0.881985 |
| bert_mid_layer |           30 |           768 | avg             |          128 |          0.0001 | adam        |      15 |   0.881985 |
| bert_lstm      |           40 |           768 | max             |           64 |          1e-05  | adam        |      15 |   0.881568 |
| bert_cnn       |           30 |           768 | max             |           64 |          0.0001 | adam        |      15 |   0.881568 |
| bert_cnn       |           30 |           768 | max             |          128 |          0.0001 | adam        |      15 |   0.881151 |
| bert_cnn       |           30 |           768 | avg             |          128 |          0.0001 | adam        |      15 |   0.8799   |
| bert_cnn       |           30 |           768 | max             |           64 |          1e-05  | adam        |      15 |   0.8799   |
| bert_mid_layer |           30 |           768 | avg             |           64 |          0.0001 | adam        |      15 |   0.8799   |
| bert_mid_layer |           30 |           768 | avg             |           64 |          1e-05  | adam        |      15 |   0.8799   |
| bert_lstm      |           40 |           768 | avg             |           64 |          1e-05  | adam        |      15 |   0.879483 |
| bert_cnn       |           40 |           768 | max             |           64 |          0.0001 | adam        |      15 |   0.879483 |
| bert_cnn       |           30 |           768 | max             |          128 |          1e-05  | adam        |      15 |   0.879483 |
| bert_cnn       |           40 |           768 | avg             |          128 |          1e-05  | adam        |      15 |   0.879483 |
| bert_lstm      |           30 |           768 | avg             |           64 |          0.0001 | adam        |      15 |   0.879066 |
| bert_lstm      |           30 |           768 | max             |          128 |          0.0001 | adam        |      15 |   0.878649 |
| bert_lstm      |           40 |           768 | avg             |          128 |          1e-05  | adam        |      15 |   0.878649 |
| bert_lstm      |           30 |           768 | max             |          128 |          1e-05  | adam        |      15 |   0.876981 |
| bert_cnn       |           30 |           768 | avg             |          128 |          1e-05  | adam        |      15 |   0.876981 |
| bert_mid_layer |           30 |           768 | avg             |          128 |          1e-05  | adam        |      15 |   0.876147 |
| bert_lstm      |           30 |           768 | max             |           64 |          0.0001 | adam        |      15 |   0.875313 |
| bert_mid_layer |           30 |           768 | avg             |          128 |          0.001  | adam        |      15 |   0.875313 |
| bert_cnn       |           40 |           768 | max             |          128 |          1e-05  | adam        |      15 |   0.874479 |
| bert_cnn       |           30 |           768 | avg             |           64 |          0.0001 | adam        |      15 |   0.873228 |
| bert_cnn       |           40 |           768 | avg             |           64 |          1e-05  | adam        |      15 |   0.873228 |
| bert_lstm      |           30 |           768 | avg             |          128 |          1e-05  | adam        |      15 |   0.871977 |
| bert_mid_layer |           30 |           768 | max             |          128 |          1e-05  | adam        |      15 |   0.871977 |
| bert_mid_layer |           40 |           768 | avg             |          128 |          0.001  | adam        |      15 |   0.87156  |
| bert_lstm      |           30 |           768 | max             |           64 |          1e-05  | adam        |      15 |   0.866555 |
| bert_mid_layer |           30 |           768 | max             |          128 |          0.001  | adam        |      15 |   0.844871 |
| bert_mid_layer |           40 |           768 | max             |           64 |          0.001  | adam        |      15 |   0.835279 |
| bert_mid_layer |           30 |           768 | avg             |           64 |          0.001  | adam        |      15 |   0.834445 |
| bert_mid_layer |           40 |           768 | avg             |           64 |          0.001  | adam        |      15 |   0.798582 |
| bert_mid_layer |           30 |           768 | max             |           64 |          0.001  | adam        |      15 |   0.788157 |
| bert           |           30 |           768 | avg             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert           |           30 |           768 | max             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert           |           30 |           768 | avg             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert           |           30 |           768 | max             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert           |           40 |           768 | avg             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert           |           40 |           768 | max             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert           |           40 |           768 | avg             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert           |           40 |           768 | max             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert_lstm      |           30 |           768 | avg             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert_lstm      |           30 |           768 | max             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert_lstm      |           30 |           768 | avg             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert_lstm      |           30 |           768 | max             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert_lstm      |           40 |           768 | avg             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert_lstm      |           40 |           768 | max             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert_lstm      |           40 |           768 | avg             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert_lstm      |           40 |           768 | max             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert_cnn       |           30 |           768 | avg             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert_cnn       |           30 |           768 | max             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert_cnn       |           30 |           768 | avg             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert_cnn       |           30 |           768 | max             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert_cnn       |           40 |           768 | avg             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert_cnn       |           40 |           768 | max             |           64 |          0.001  | adam        |      15 |   0.664721 |
| bert_cnn       |           40 |           768 | avg             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert_cnn       |           40 |           768 | max             |          128 |          0.001  | adam        |      15 |   0.664721 |
| bert_mid_layer |           40 |           768 | max             |          128 |          0.001  | adam        |      15 |   0.362802 |