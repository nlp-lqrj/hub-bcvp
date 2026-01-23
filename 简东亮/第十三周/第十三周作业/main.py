导入 手电筒
导入 操作系统
导入 随机数
导入 操作系统
导入 numpy库 并将其命名为 np
导入 torch.nn作为nn  
导入 日志
from  config  import  Config
from  model  import  TorchModel , choose_optimizer
从 evaluate 导入 Evaluator
from  loader  import  load_data
from  peft  import  get_peft_model , LoraConfig , \
    PromptTuningConfig、PrefixTuningConfig、PromptEncoderConfig


#[调试、信息、警告、错误、严重]
logging.basicConfig ( level = logging.INFO , format = '%(asctime)s - %( name )s - %(levelname)s - %(message) s ' )  
logger  =  logging.getLogger ( __ name__ )

"""
模型训练主程序
"""


seed  =  Config [ "seed" ]
random.seed（种子）
np.random.seed（种子）
torch.manual_seed（种子）
torch.cuda.manual_seed_all（种子）



def  main ( config ):
    #创建保存模型的目录
    如果 不是 os.path.isdir ( config [ " model_path " ] ) ：
        os.mkdir ( config [ "model_path " ] )
    #加载训练数据
    train_data  =  load_data ( config [ "train_data_path" ], config )
    #加载模型
    from  model  import  get_model
    model  =  get_model ()

    #大型模特压力策略
    tuning_tactics  =  config [ "tuning_tactics" ]
    如果 tuning_战术 ==  “lora_tuning”：
        peft_config  =  LoraConfig (
            r = 8，
            lora_alpha = 32，
            lora_dropout = 0.1，
            target_modules = [ "query" , "key" , "value" ]
        ）
    elif  tuning_tactics  ==  "p_tuning" :
        # TOKEN_CLS：Token分类（序列标签），如NER、词性标签
        # SEQ_CLS：序列分类（句子分类），情感分析、文本分类
        peft_config  =  PromptEncoderConfig ( task_type = "TOKEN_CLS" , num_virtual_tokens = 10 )
    elif  tuning_tactics  ==  "prompt_tuning" :
        peft_config  =  PromptTuningConfig ( task_type = "TOKEN_CLS" , num_virtual_tokens = 10 )
    elif  tuning_tactics  ==  "prefix_tuning" :
        peft_config  =  PrefixTuningConfig ( task_type = "TOKEN_CLS" , num_virtual_tokens = 10 )


    model  =  get_peft_model ( model , peft_config )
    # print(model.state_dict().keys())

    如果 tuning_tactics  ==  "lora_tuning" :
        # lora配置会冻结原始模型中所有层的权限重，不允许其反传轻微
        # 但实际上我们希望最后一个线性层照常训练，只是 bert 部分被冻结，所以需要手动设置
        # 对于TokenClassification，分类层名称可能是classifier
        尝试：
            for  param  in  model.get_submodule ( " model" ). get_submodule ( "classifier" ). parameters ( ):
                param.requires_grad = True​​  
        除了：
            # 如果分类器不存在，尝试其他可能的名称
            经过

    # 标识是否使用gpu
    cuda_flag  =  torch.cuda.is_available ( )​​​
    如果 cuda_flag：
        记录器。info ( "gpu可以使用，迁移模型至gpu" )
        model  =  model.cuda ( )​

    #加载优化器
    optimizer  =  choose_optimizer ( config , model )
    #加载效果测试类
    评估器 = 评估器（配置，模型，日志记录器）
    # 训练
    for  epoch  in  range ( config [ "epoch" ]):
        epoch  +=  1
        模型.训练()
        logger.info ( " epoch %d begin " % epoch )  
        train_loss  = []
        for  index , batch_data  in  enumerate ( train_data ):
            如果 cuda_flag：
                batch_data  = [ d . cuda () for  d  in  batch_data ]

            优化器.zero_grad ( )
            input_ids、attendance_mask、labels  =  batch_data   # NER 任务：input_ids、attendance_mask、labels
            outputs  =  model ( input_ids = input_ids , attention_mask = attention_mask )
            # 处理PEFT模型可能返回元组的情况
            如果 isinstance ( outputs , tuple ):
                logits  =  outputs [ 0 ]   #如果是元组，取第一个元素
            别的：
                logits  =  outputs.logits #形状：[batch_size ,   seq_len, num_labels]

            # 计算损失，忽略-100标签（padding和subword token）
            loss_fct  =  nn.CrossEntropyLoss ( ignore_index = -100 )​​​
            loss  =  loss_fct ( logits.view ( -1 , logits.size ( -1 ) ) , labels.view ( -1 ) )​​​​​​
            损失.向后()
            优化器.步骤()

            train_loss.append ( loss.item ( ) )​​
            如果 index  %  int ( len ( train_data ) /  2 ) ==  0 :
                logger.info ( "批次损失 %f " %损失)  
        logger.info ( " epoch 平均损失：% f " % np.mean ( train_loss ) )  
        acc  =  evaluator.eval ( epoch )​​
    model_path  =  os.path.join ( config [ "model_path" ] , " % s.pth " % tuning_tactics )  
    save_tunable_parameters ( model , model_path )    #保存模型权重
    返回 账户

def  save_tunable_parameters ( model , path ):
    saved_pa​​rams  = {
        k : v .到（“cpu”）
        for  k , v  in  model.named_pa ​​rameters ( )
        如果 v.requires_grad
    }
    torch.save ( saved_pa​​)公羊，路径）


如果 __name__  ==  "__main__" :
    主要（配置）
