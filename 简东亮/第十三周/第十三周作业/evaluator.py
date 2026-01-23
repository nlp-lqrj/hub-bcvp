导入 手电筒
因 雷
导入 numpy库 并将其命名为 np
from  collections  import  defaultdict
from  loader  import  load_data

"""
模型效果测试 - 实体评估
"""

类别 评估器：
    def  __init__ ( self , config , model , logger ):
        self.config =配置  
        self.model =模型  
        self.logger = logger  
        self.valid_data = load_data  ( config [ "valid_data_path " ] , config , shuffle = False ) 
        self.index_to_label = { "B- LOCATION "  : 0 ,
                               “B组织”：1，
                               “B-PERSON”：2，
                               “B-TIME”：3，
                               "I-LOCATION" : 4 ,
                               “I-组织”：5，
                               “I-PERSON”：6，
                               “I-TIME”：7，
                               "O" : 8 }

    def  eval ( self , epoch ):
        自己。记录器。info ( "开始测试第%d轮模型效果："  %  epoch )
        self.stats_dict = { " LOCATION  " : defaultdict ( int ),
                           "TIME" : defaultdict ( int ),
                           “PERSON”：defaultdict（int），
                           "ORGANIZATION" : defaultdict ( int )}
        self.model.eval ( )

        数据 集=  self.valid_data.dataset
        sentence_idx  =  0

        for  batch_idx , batch_data  in  enumerate ( self.valid_data )）：
            # 获取当前batch对应的原始句子
            batch_size  =  batch_data [ 0 ] .size ( 0 )
            sentences  =  dataset.sentences [ sentence_idx : sentence_idx + batch_size ]​  
            sentence_idx  +=  batch_size

            如果 torch.cuda.is_available ( ) :
                batch_data  = [ d . cuda () for  d  in  batch_data ]
            input_ids、attendance_mask、labels  =  batch_data   # NER 任务：input_ids、attendance_mask、labels

            使用 torch.no_grad ( ) ：
                outputs  =  self.model ( input_ids = input_ids , attention_mask = attendance_mask )
                # 处理PEFT模型可能返回元组的情况
                如果 isinstance（输出，元组）：
                    pred_results  =  outputs [ 0 ]    #如果是元组，取第一个元素
                另外：
                    pred_results  =  outputs.logits #形状：[batch_size, seq_len, num_labels ]  

            # 获取预测标签
            pred_labels  =  torch.argmax ( pred_results , dim = -1 ) # [batch_size ,    seq_len ]

            self.write_stats ( labels , pred_labels , attention_mask , sentences , input_ids , batch_idx )

        acc  =  self.show_stats（）
        返回 账户

    def  write_stats ( self , labels , pred_labels , attention_mask , sentences , input_ids , batch_idx = 0 ):
        """计算实体级别的统计信息"""
        batch_size  =  labels.size ( 0 )

        for  i  in  range ( batch_size ):
            如果 i  >=  len (句子):
                休息时间

            Sentence  =  Sentences [ i ]    # 原始字符列表
            true_label_seq  =  labels [ i ] .cpu (). detach (). tolist ()
            pred_label_seq  =  pred_labels [ i ] .cpu (). detach (). tolist ()
            mask  =  attention_mask [ i ] .cpu (). detach (). tolist ()

            # 将BERT代币级别的标签映射回字符级别
            # 需要处理子词标记和特殊标记
            true_char_labels  =  self.map_token_labels_to_chars (
                true_label_seq、sentence、input_ids [ i ]、mask
            ）
            pred_char_labels  =  self.map_token_labels_to_chars (
                pred_label_seq、sentence、input_ids [ i ]、mask
            ）

            #解码实体
            true_entities  =  self.decode ( sentence , true_char_labels )
            pred_entities  =  self.decode ( sentence , pred_char_labels )

            # 调试信息（强制显示前5个样本，用于诊断问题）
            如果 batch_idx  ==  0 且 i  <  5：
                has_entity  =  any ( len ( true_entities [ key ]) >  0  for  key  in [ "PERSON" , "LOCATION" , "TIME" , "ORGANIZATION" ])
                info ( f "样本{ batch_idx * batch_size + i + 1 } -长度: { len ( Sentence ) } , 是否有实体: { has_entity } " )      
                自己。记录器。info ( f" 句子: { ' ' . join ( [: 50 ]) } ..." )
                info ( f " 字符等级真实标签(前 50) : { true_char_labels [ : 50 ] } " )
                自己。记录器。信息( f" 字符级别预测标签(前 50): { pred_char_labels [: 50 ] } " )
                自己。记录器。info ( f" 真实实体: { dict ( true_entities ) } " )
                自己。记录器。info ( f" 预测实体: { dict ( pred_entities ) } " )
                #显示token级别的标签（用于调试）
                non_o_true  = [ idx  for  idx , lbl  in  enumerate ( true_label_seq [: 80 ]) if  lbl  !=  - 100  and  lbl  !=  8 ]
                non_o_pred  = [ idx  for  idx , lbl  in  enumerate ( pred_label_seq [: 80 ]) if  lbl  !=  - 100  and  lbl  !=  8 ]
                如果 non_o_true：
                    info ( f " Token等级真实标签(非O位置，前20个): { [ ( idx , true_label_seq [ idx ] ) for idx in non_o_true [ : 20 ]] } " )   
                另外：
                    info ( f " Token等级真实标签(前30个) : { true_label_seq [ : 30 ] } " )
                如果 non_o_pred：
                    info ( f"代币级别预测标签(非O位置，前20个): { [ ( idx , pred_label_seq [ idx ] ) for idx in non_o_pred [ : 20 ]] } " )   
                另外：
                    自己。记录器。信息( f" 代币级别预测标签(前30个): { pred_label_seq [: 30 ] } " )
                self.logger.info ( " " )

            # 统计每个实体类型
            for  key  in [ "PERSON" , "LOCATION" , "TIME" , "ORGANIZATION" ]:
                self.stats_dict [ key ][ "正确识别" ] += len ( [ ent for ent in pred_entities [ key ] if ent in true_entities [ key] ] )        
                stats_dict [ key ][ "样本实体数" ] + = len ( true_entities [ key ]) 
                stats_dict [ key ][ "识别出实体数" ] += len ( pred_entities [ key ] ) 
        返回

    def  map_token_labels_to_chars ( self , token_labels , sentence , input_ids , mask ):
        """将 BERT 代币级别的标签映射回字符级别"""
        如果 self.config["model_type" ] ! =  "bert " ：
            #非BERT模型，直接返回
            返回 token_labels [: len (句子)]

        # 获取分词器
        tokenizer  =  self.valid_data.dataset.tokenizer

        # 对于BERT，需要处理subword token
        # 在loader中，每个角色的第一个子词标记都有标签，其他子词标记标签为-100
        char_labels  = []
        char_idx  =  0

        # 跳过[CLS]，从索引1开始
        for  token_idx  in  range ( 1 , len ( token_labels )):
            如果 token_idx  >=  len ( mask )或 mask [ token_idx ] ==  0：
                休息时间

            # 检查是否是特殊令牌([SEP], [PAD])
            token_id  =  input_ids [ token_idx ] .item ( ) if  isinstance ( input_ids , torch.Tensor ) else input_ids [ token_idx ] 
            如果 token_id  ==  tokenizer.sep_token_id或token_id  == tokenizer.pad_token_id：   
                休息时间

            # 如果标签是-100，跳过（这是子词标记的后续部分，在loader中已标记）
            如果 token_labels [ token_idx ] ==  -100 ：
                继续

            #非-100的标签对应一个字符的第一个子词标记，取这个标签
            如果 char_idx  <  len (句子):
                char_labels.append ( token_labels [ token_idx]] )
                char_idx  +=  1
            另外：
                休息时间

        #确定长度匹配（如果句子被截断，用O标签填充）
        当 len ( char_labels ) <  len (句子):
            字符标签。追加( 8 )    #O标签

        返回 char_labels [: len (句子)]

    def  show_stats ( self ):
        """显示统计结果"""
        F1_scores  = []
        for  key  in [ "PERSON" , "LOCATION" , "TIME" , "ORGANIZATION" ]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 识别率 = 识别出的正确实体数 / 样本的实体数
            stats_dict  [ key ][ "正确识别" ] / ( 1e -5 + self . stats_dict [ key ] [  "识别出实体数" ])  
            stats_dict  [ key ][ "正确识别" ] / ( 1e -5 + self . stats_dict [ key ]  [ "样本回忆实体数" ])  
            F1  = ( 2  * 准确率 * 响应率) / (准确率 + 响应率 +  1e-5 )
            F1_scores.append（F1）
            info ( " % s类实体，准确率：%f,识别率: %f, F1: %f " % ( key , precision , recall , F1 )) 

        self.logger.info ( " Macro -F1: % f " % np.mean ( F1_scores ) )  

        correct_pred  =  sum ([ self.stats_dict [ key ][ "正确识别" ] for key in [ " PERSON " , "LOCATION" , "TIME" , "ORGANIZATION" ]])  
        Total_pred  =  sum ([ self . stats_dict [ key ][ "识别出实体数" ] for  key  in [ "PERSON" , "LOCATION" , "TIME" , "ORGANIZATION" ]])
        true_enti  =  sum ([ self . stats_dict [ key ][ "样本实体数" ] for  key  in [ "PERSON" , "LOCATION" , "TIME" , "ORGANIZATION" ]])
        微精度 = 正确预测值 /（总预测值 +  1e-5）
        微回忆率 = 正确预测值 /（真实值 +  1e-5）
        micro_f1  = ( 2  *  micro_precision  *  micro_recall ) / ( micro_precision  +  micro_recall  +  1e-5 )
        self.logger.info ( " Micro-F1 % f " % micro_f1 )  
        self.logger.info ( " -------------------- " )
        返回 micro_f1

    def  decode ( self , sentence , labels ):
        """
        解码标签序列，提取实体
        labels: 标签索引列表，如 [8, 8, 0, 4, 4, 8, ...]
        返回：{“地点”：[...]，“人物”：[...]，...}
        """
        # 将标签索引转换为字符串，用于正则匹配
        # B-LOCATION=0, I-LOCATION=4 -> "0" + "4"*
        # B-ORGANIZATION=1, I-ORGANIZATION=5 -> "1" + "5"*
        # B-PERSON=2，I-PERSON=6 -> "2" + "6"*
        # B-TIME=3，I-TIME=7 -> "3" + "7"*
        labels_str  =  "" . join ([ str ( x ) for  x  in  labels [: len ( sentence )]])
        results  =  defaultdict ( list )

        # 匹配位置：B-位置(0) + I-位置(4)*
        # 正则表达式：0后面跟一个或多个4
        for  match  in  re.finditer ( "0 [4]*" , labels_str ) :
            s，e  =  match.span（）
            如果 s  <  len（句子）：
                results [ "LOCATION" ]. append ( "" . join ( sentence [ s : min ( e , len ( sentence ))]))

        # 合并组织：B组织(1) + I组织(5)*
        # 正则表达式：1后面跟一个或多个5
        for  match  in  re.finditer ( "1[5]* " , labels_str ) :
            s，e  =  match.span（）
            如果 s  <  len（句子）：
                results [ "ORGANIZATION" ]. append ( "" . join ( sentence [ s : min ( e , len ( sentence ))]))

        # 匹配PERSON: B-PERSON(2) + I-PERSON(6)*
        # 正则表达式：2后面跟一个或多个6
        for  match  in  re.finditer ( "2[6]* " , labels_str ) :
            s，e  =  match.span（）
            如果 s  <  len（句子）：
                results [ "PERSON" ]. append ( "" . join ( sentence [ s : min ( e , len ( sentence ))]))

        # 匹配TIME: B-TIME(3) + I-TIME(7)*
        # 正则表达式：3后面跟一个或多个7
        for  match  in  re.finditer ( "3[7]* " , labels_str ) :
            s，e  =  match.span（）
            如果 s  <  len（句子）：
                results [ "TIME" ]. append ( "" . join ( sentence [ s : min ( e , len ( sentence ))]))

        返回 结果
