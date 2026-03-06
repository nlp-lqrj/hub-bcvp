'''
任务型多轮对话系统
读取场景脚本完成多轮对话

编程建议：
1.先搭建出整体框架
2.先写测试用例，确定使用方法
3.每个方法不要超过20行
4.变量命名和方法命名尽量标准
'''

import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.load()
        self.last_response = ""  # 记录上一次回复
        self.last_memory_state = {}  # 记录上一个状态
    
    def load(self):
        self.all_node_info = {}
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_templet("slot_fitting_templet.xlsx")


    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['childnode'] = [scenario_name + "_" + x for x in node['childnode']]
            

    def load_slot_templet(self, file):
        self.slot_templet = pd.read_excel(file)
        #三列：slot, query, values
        self.slot_info = {}
        #逐行读取，slot为key，query和values为value
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = str(values)  # 确保转换为字符串
      
    def nlu(self, memory):
        # 对"没听清"特殊处理，不进行意图判断
        if memory.get('query', '').strip() == "没听清":
            return memory
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        #意图识别，匹配当前可以访问的节点
        query = memory['query']
        # 如果是"没听清"，则跳过意图判断
        if query.strip() == "没听清":
            return memory
        
        max_score = -1
        hit_node = None
        for node in memory["available_nodes"]:
            score = self.calucate_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory


    def calucate_node_score(self, query, node):
        #节点意图打分，算和intent相似度
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calucate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score
    
    def calucate_sentence_score(self, query, sentence):
        #两个字符串做文本相似度计算。jaccard距离计算相似度
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)


    def slot_filling(self, memory):
        #槽位填充
        # 如果是"没听清"，则不进行槽位填充
        if memory.get('query', '').strip() == "没听清":
            return memory
            
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        query = memory['query']
        
        for slot in node_info.get('slot', []):
            if slot not in memory:
                if slot in self.slot_info:
                    slot_values_pattern = self.slot_info[slot]["values"]
                    # 避免NaN值导致的问题
                    if slot_values_pattern and slot_values_pattern != "nan":
                        try:
                            match_result = re.search(slot_values_pattern, query)
                            if match_result:
                                memory[slot] = match_result.group()
                        except re.error:
                            # 如果正则表达式有问题，则尝试简单的子串匹配
                            if slot_values_pattern.lower() in query.lower():
                                memory[slot] = slot_values_pattern
                else:
                    # 如果槽位不在模板中，尝试更通用的方法
                    # 特别针对数值类型的槽位
                    if "#分期付款期数#" == slot and re.search(r'\d+', query):
                        number_match = re.search(r'\d+', query)
                        memory[slot] = number_match.group()
                    elif "#服装尺寸#" == slot and re.search(r'[SMLXsml]+', query):
                        size_match = re.search(r'[SMLXsml]+', query)
                        memory[slot] = size_match.group().upper()
        return memory

    def dst(self, memory):
        # 如果是"没听清"，则不进行状态跟踪
        if memory.get('query', '').strip() == "没听清":
            return memory
            
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        slot = node_info.get('slot', [])
        for s in slot:
            if s not in memory:
                memory["require_slot"] = s
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        # 如果是"没听清"，则不进行策略优化
        if memory.get('query', '').strip() == "没听清":
            return memory
            
        if memory["require_slot"] is None:
            #没有需要填充的槽位
            memory["policy"] = "reply"
            # self.take_action(memory)
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            #有欠缺的槽位
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]] #停留在当前节点
        return memory

    def nlg(self, memory):
        # 如果是"没听清"，则不进行语言生成
        if memory.get('query', '').strip() == "没听清":
            return memory
            
        #根据policy执行反问或回答
        if memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
        else:
            #policy == "request"
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]
        return memory

    def fill_in_slot(self, template, memory):
        node = memory["hit_node"]
        node_info = self.all_node_info[node]
        for slot in node_info.get("slot", []):
            if slot in memory:  # 确保槽位已被填充
                template = template.replace(slot, memory[slot])
        return template

    def run(self, query, memory):
        '''
        query: 用户输入
        memory: 用户状态
        '''
        # 检查是否是要求重播的指令
        if query.strip() == "没听清":
            # 恢复到上一个状态并重播上次的回复
            if self.last_memory_state:
                memory.clear()
                memory.update(self.last_memory_state.copy())
            return memory
        
        # 保存当前状态作为上一状态
        self.last_memory_state = memory.copy()
        
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        
        # 保存本次回复作为下次可能的历史记录
        if "response" in memory:
            self.last_response = memory["response"]
        
        return memory
        

if __name__ == '__main__':
    ds = DialogueSystem()
    # print(ds.all_node_info)
    print(ds.slot_info)

    memory = {"available_nodes":["scenario-买衣服_node1","scenario-看电影_node1"]} #用户状态
    while True:
        query = input("请输入：")
        # query = "你好"    
        memory = ds.run(query, memory)
        print(memory)
        print()
        response = memory['response']
        print(response)
        print("===========")
