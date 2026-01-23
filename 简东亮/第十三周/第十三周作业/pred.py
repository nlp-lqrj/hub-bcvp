导入 手电筒
导入 日志
from  model  import  get_model
from  peft  import  get_peft_model , LoraConfig , PromptTuningConfig , PrefixTuningConfig , PromptEncoderConfig

从 evaluate 导入 Evaluator
from  config  import  Config


logging.basicConfig ( level = logging.INFO , format = '%(asctime)s - %( name )s - %(levelname)s - %(message) s ' )  
logger  =  logging.getLogger ( __ name__ )

#大型模特压力策略
tuning_tactics  =  Config [ "tuning_tactics" ]

print ( "正在使用 %s" % adjustment_tropics )

如果 tuning_战术 ==  “lora_tuning”：
    peft_config  =  LoraConfig (
        r = 8，
        lora_alpha = 32，
        lora_dropout = 0.1，
        target_modules = [ "query" , "key" , "value" ]
    ）
elif  tuning_tactics  ==  "p_tuning" :
    peft_config  =  PromptEncoderConfig ( task_type = "TOKEN_CLS" , num_virtual_tokens = 10 )
elif  tuning_tactics  ==  "prompt_tuning" :
    peft_config  =  PromptTuningConfig ( task_type = "TOKEN_CLS" , num_virtual_tokens = 10 )
elif  tuning_tactics  ==  "prefix_tuning" :
    peft_config  =  PrefixTuningConfig ( task_type = "TOKEN_CLS" , num_virtual_tokens = 10 )

#重建模型
model  =  get_model ()
# print(model.state_dict().keys())
# print("========================")

model  =  get_peft_model ( model , peft_config )
# print(model.state_dict().keys())
# print("========================")

state_dict  =  model.state_dict（）

#将账户部分权重加载
如果 tuning_战术 ==  “lora_tuning”：
    loaded_weight  =  torch.load ( '输出'/lora_tuning.pth' )
elif  tuning_tactics  ==  "p_tuning" :
    loaded_weight  =  torch.load ( ' output/p_tuning.pth ' )
elif  tuning_tactics  ==  "prompt_tuning" :
    loaded_weight  =  torch.load ( ' output/prompt_tuning.pth ' )
elif  tuning_tactics  ==  "prefix_tuning" :
    loaded_weight  =  torch.load ( '输出'/prefix_tuning.pth' )

print ( loaded_weight.keys ( ) )
对于 loaded_weight.items()中的每个键和值 ： 
    print ( key , value.shape )
state_dict.update ( loaded_weight )

#权重更新后重新加载到模型
model.load_state_dict ( state_dict )

#进行一次测试
模型 =  model.cuda（）
评估器 = 评估器（配置，模型，日志记录器）
评估器.评估( 0 )
