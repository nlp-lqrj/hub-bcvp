导入 torch.nn作为nn  
from  config  import  Config
from  transformers  import  AutoTokenizer , AutoModelForTokenClassification , AutoModel
来自 torch.optim导入 Adam、SGD 

def  get_model ():
    """获取模型，延迟加载"""
    返回 来自预训练的 AutoModelForTokenClassification（
        配置[ "pretrain_model_path" ]，
        num_labels = Config [ "class_num" ]
    ）

# 为了兼容原有代码，保留TorchModel指标
火炬模型 = 无


def  choose_optimizer ( config , model ):
    优化器 = 配置[ "优化器" ]
    learning_rate  =  config [ "learning_rate" ]
    如果 优化器 ==  "adam"：
        返回 Adam (模型.参数()，lr =学习率)
    elif  optimizer  ==  "sgd" :
        返回 SGD (模型.参数()，lr =学习率)
