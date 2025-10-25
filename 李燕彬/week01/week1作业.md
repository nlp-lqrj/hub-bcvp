## 20251012-week01-第一周作业

- 作业1: 准备开发环境，参考[预习资料](https://note.youdao.com/s/BAcOYrOB)
- 作业2: 配置github，并提交个人文件夹(作业提交用途) 参考[提交代码流程](https://note.youdao.com/s/KXGBGt8D)

### 作业1内容
#### 以下是在mac系统中，安装深度学习环境后的，软件版本输出信息

```
(base) MacBook-Pro:~ felix$ which conda
/opt/miniconda3/bin/conda
(base) MacBook-Pro:~ felix$ conda env list

# conda environments:
#
base                 * /opt/miniconda3
py312                  /opt/miniconda3/envs/py312

(base) MacBook-Pro:~ felix$ conda activate py312
(py312) MacBook-Pro:~ felix$ pip list | egrep 'numpy|pandas|matplotlib|learn|peft|gensim|transformers'
gensim                    4.3.3
matplotlib                3.9.2
matplotlib-inline         0.1.7
numpy                     1.26.4
pandas                    2.2.2
peft                      0.15.0
scikit-learn              1.5.1
transformers              4.49.0
```

#### 引用numpy和matplotlib.pyplot模块实现sin函数的绘图
```
(py312) MacBook-Pro:~ felix$ python3
Python 3.12.11 | packaged by Anaconda, Inc. | (main, Jun  5 2025, 08:06:15) [Clang 14.0.6 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> x = np.arange(0, 6, 0.1)
>>> y = np.sin(x)
>>> print(y)
[ 0.          0.09983342  0.19866933  0.29552021  0.38941834  0.47942554
  0.56464247  0.64421769  0.71735609  0.78332691  0.84147098  0.89120736
  0.93203909  0.96355819  0.98544973  0.99749499  0.9995736   0.99166481
  0.97384763  0.94630009  0.90929743  0.86320937  0.8084964   0.74570521
  0.67546318  0.59847214  0.51550137  0.42737988  0.33498815  0.23924933
  0.14112001  0.04158066 -0.05837414 -0.15774569 -0.2555411  -0.35078323
 -0.44252044 -0.52983614 -0.61185789 -0.68776616 -0.7568025  -0.81827711
 -0.87157577 -0.91616594 -0.95160207 -0.97753012 -0.993691   -0.99992326
 -0.99616461 -0.98245261 -0.95892427 -0.92581468 -0.88345466 -0.83226744
 -0.77276449 -0.70554033 -0.63126664 -0.55068554 -0.46460218 -0.37387666]
>>> import matplotlib.pyplot as plt
>>> plt.plot(x, y)
[<matplotlib.lines.Line2D object at 0x11f5149e0>]
>>> plt.show()
2025-10-16 18:13:08.148 python3[75755:5644139] The class 'NSSavePanel' overrides the method identifier.  This method is implemented by class 'NSWindow'
```
##### 输出图片内容如下：

![sin.png](sin.png)

### 作业2内容
#### 看到此处提交的内容，则说明配置github成功

### 备注：[NLP学习笔记-Week01](https://share.note.youdao.com/s/9kehADt4)