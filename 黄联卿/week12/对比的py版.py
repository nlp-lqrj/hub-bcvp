"""
模型名称	    位置编码   transformer结构    多头机制	     ff层设计	    归一化层选择	          激活函数	       是否使用bias
DeepseekV3	RoPE	      串行	        deepseek MLA	   gated形式	  RMSnorm/pre norm	    SiLU	           无bias
Qwen-7B	    RoPE	      串行	        传统方式	         传统方式	    RMSnorm/pre norm	    SiLU	           无bias
Mixtral	    RoPE	      串行	        grouped query	   gated形式	  RMSnorm/pre norm	    SiLU	           无bias
DBRX	      RoPE	      串行	        grouped query	   gated形式	  LayerNorm/pre norm	  配置文件未指定	   无bias
gemma	      RoPE	      串行	        grouped query	   传统方式	    RMSnorm/pre norm	    gelu	           无bias
"""
