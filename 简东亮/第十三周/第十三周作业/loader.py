导入 json
因 雷
导入 操作系统
导入 手电筒
导入 numpy库 并将其命名为 np
from  torch.utils.data import Dataset , DataLoader  
from  transformers  import  BertTokenizer
"""
数据加载 - NER任务
"""


类 数据生成器：
    def  __init__ ( self , data_path , config ):
        self.config =配置  
        self.path =数据路径  
        self.index_to_label = { "B- LOCATION "  : 0 ,
                               “B组织”：1，
                               “B-PERSON”：2，
                               “B-TIME”：3，
                               "I-LOCATION" : 4 ,
                               “I-组织”：5，
                               “I-PERSON”：6，
                               “I-TIME”：7，
                               “O”：8
                               }
        self.label_to_index = dict ( ( y , x ) for x , y in self.index_to_label.items ( ) )     
        self.config [ " class_num " ] = len ( self.index_to_label ) 
        如果 self.config[ " model_type" ] == " bert"： 
            self.tokenizer = BertTokenizer.from_pretrained  ( config [ " pretrain_model_path " ] ) 
        self.vocab = load_vocab  ( config [ " vocab_path" ] ) 
        self.config [ " vocab_size " ] = len ( self.vocab ) 
        自身加载（）


    def  load ( self ):
        """加载NER数据，格式为每行一个令牌和标签，空行分隔句子"""
        self.data = [  ]
        Sentence = []    #保存句子原始（字符列表） 
        tokens  = []
        标签 = []

        with  open ( self.path , encoding = " utf8 " ) as f : 
            Forf 中的每行 ： 
                line  =  line.strip（）
                if  not  line :    # 空行表示句子结束
                    如果 代币：
                        self.process_sentence ( tokens , labels )
                        tokens  = []
                        标签 = []
                另外：
                    parts  =  line.split（）
                    如果 len (部分) >=  2：
                        token  =  parts [ 0 ]
                        标签 = 部分[ 1 ]
                        tokens.append（令牌）
                        labels.append ( label )​​

            # 处理最后一个句子（如果文件补充没有空行）
            如果 tokens：
                self.process_sentence ( tokens , labels )​​
        返回

    def  process_sentence ( self , tokens , labels ):
        """处理一个句子，将tokens和labels编码为模型输入格式"""
        # 保存原始句子（字符列表）和对应的原始标签
        self.sentences.append ( tokens.copy ( ) )​​​​
        # 将代币组合成文本
        text  =  "" . join ( tokens )

        # 使用tokenizer编码
        如果 self.config [ " model_type" ] == " bert"： 
            # 使用encode_plus获取更多信息
            encoded  =  self.tokenizer.encode_plus (​​​​
                文本，
                max_length = self.config [ "max_length " ] ,
                padding = "max_length" ,
                截断= True，
                return_tensors = "pt" ,
                return_offsets_mapping = False
            ）
            input_ids  =  encoded [ "input_ids" ]. squeeze ( 0 )
            attention_mask  =  encoded [ "attention_mask" ]. squeeze ( 0 )

            # 对齐标签：对于中文BERT，需要将字符级别的标签对齐到令牌级别
            # 关键问题：逐个字符tokenize和整个句子tokenize的结果可能不一致
            # 解决方法：使用整个句子的分词结果，然后通过字符位置对齐
            aligned_labels  = []

            # 跳过[CLS]
            aligned_labels.append ( -100 )​​​

            # 使用整个句子的 tokenize 结果来对齐
            # 对于中文BERT，通常是一个字符对应一个token（或少数几个subword token）
            # 我们需要找到每个token对应的字符位置
            char_pos  =  0   # 当前字符位置

            # 逐个字符处理，找到对应的token
            for  char_idx , char  in  enumerate ( tokens ):
                如果 char_idx  >=  len ( labels ):
                    休息

                # tokenize当前字符（用于确定需要多少个token）
                char_tokens  =  self.tokenizer.encode ( char , add_special_tokens = False )​​​​
                label_str  =  labels [ char_idx ]
                label_id  =  self.label_to_index.get ( label_str , 8 )​​​​

                # 第一个子词标记使用原标签
                如果 len ( char_tokens ) >  0：
                    aligned_labels.append ( label_id )​​
                    # 如果有多个subword token，其余设为-100
                    for  _  in  range ( len ( char_tokens ) -  1 ):
                        aligned_labels.append ( -100 )​​​

                # 检查是否超过最大长度（-1 for [SEP]）
                如果 len ( aligned_labels ) >=  self.config [ " max_length " ] -  1 :
                    休息

            #添加[SEP]和填充
            aligned_labels.append ( -100 ) # [ SEP   ]
            while  len ( aligned_labels ) <  self.config [ "max_length " ] :
                aligned_labels.append ( -100 )   #填充​​

            #确定长度一致
            aligned_labels  =  aligned_labels [: self.config [ " max_length" ] ]

            labels_tensor  =  torch.LongTensor ( aligned_labels )​​
        别的：
            # 非BERT模型的处理
            input_ids  =  self.encode_sentence ( text )​​
            input_ids  =  torch.LongTensor ( input_ids )​​
            attention_mask  =  torch.ones_like ( input_ids )​​

            # 对齐标签
            aligned_labels  = []
            for  i , label_str  in  enumerate ( labels ):
                aligned_labels.append ( self.label_to_index.get ( label_str , 8 ) )​​​​​
            # 内边距标签
            while  len ( aligned_labels ) <  self.config [ "max_length " ] :
                aligned_labels.append ( 8 ) #   O​
            aligned_labels  =  aligned_labels [: self.config [ " max_length" ] ]
            labels_tensor  =  torch.LongTensor ( aligned_labels )​​

        self.data.append ( [ input_ids , attention_mask , labels_tensor ] )​​

    def  encode_sentence ( self , text ):
        input_id  = []
        对于 文本中的每个字符 ： 
            input_id.append ( self.vocab.get ( char , self.vocab [ " [ UNK ] " ] ) )​​
        input_id  =  self.padding ( input_id )​​
        返回 input_id

    #补齐或中断输入的序列，可以在一个批处理内损坏
    def  padding ( self , input_id ):
        input_id  =  input_id [: self.config [ " max_length" ] ]
        input_id  += [ 0 ] * ( self . config [ "max_length" ] -  len ( input_id ))
        返回 input_id

    def  __len__ ( self ):
        返回 len ( self.data )​​

    def  __getitem__ (自身,索引):
        返回 self.data [ index ]​​

def  load_vocab ( vocab_path ):
    token_dict  = {}
    with  open ( vocab_path , encoding = "utf8" ) as  f :
        对于 索引，行在 enumerate  ( f ) :
            token  =  line.strip ( )​
            token_dict [ token ] =  index  +  1   #0 移动padding位置，所以从1开始
    返回 token_dict


#用torch自带的DataLoader类封装数据
def  load_data ( data_path , config , shuffle = True ):
    dg  = 数据生成器（数据路径，配置）
    dl  =  DataLoader ( dg , batch_size = config [ "batch_size" ], shuffle = shuffle )
    返回 dl

如果 __name__  ==  "__main__" :
    from  config  import  Config
    dg  =  DataGenerator ( "valid_tag_news.json" , Config )
    print ( dg [ 1 ])
