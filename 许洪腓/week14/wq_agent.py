import os
import json
import requests
import pandas as pd 
from openai import OpenAI

# 全局变量存储会话，解决Session对象无法JSON序列化的问题
global_session = None

def load_credential():
    try:
        with open(r'credential.json', 'r', encoding='utf8') as f:
            credential = json.load(f)  
        # 返回字典而非JSON字符串，方便后续使用
        return credential
    except FileNotFoundError:
        return {"error": "credential.json文件未找到"}
    except json.JSONDecodeError:
        return {"error": "credential.json文件格式错误"}

# ==================== 工具函数定义 ====================
def login(username: str = None, password: str = None):
    global global_session
    if not username or not password:
        return {"error": "用户名或密码不能为空"}
    
    try:
        # Create a session to persistently store the headers
        s = requests.Session()
        # Save credentials into session
        s.auth = (username, password)
        # Send a POST request to the /authentication API
        response = s.post('https://api.worldquantbrain.com/authentication')
        response.raise_for_status()  # 抛出HTTP错误
        global_session = s
        return {"status": "success", "message": "登录成功", "content": str(response.content)[:200]}
    except requests.exceptions.RequestException as e:
        return {"error": f"登录失败：{str(e)}"}

def get_datasets(
    instrument_type: str = 'EQUITY',
    region: str = 'USA',
    delay: int = 1,
    universe: str = 'TOP3000'
):
    # 从全局变量获取会话，不再通过参数传递
    s = global_session
    if not s:
        return {"error": "请先登录获取会话"}
    
    try:
        url = "https://api.worldquantbrain.com/data-sets?" +\
            f"instrumentType={instrument_type}&region={region}&delay={str(delay)}&universe={universe}"
        result = s.get(url)
        result.raise_for_status()
        datasets_df = pd.DataFrame(result.json()['results'])
        # 将DataFrame转为JSON字符串返回（方便序列化）
        return {"status": "success", "data": datasets_df.to_dict('records')}
    except requests.exceptions.RequestException as e:
        return {"error": f"获取数据集失败：{str(e)}"}
    except Exception as e:
        return {"error": f"数据处理失败：{str(e)}"}

def get_datafields(
    instrument_type: str = 'EQUITY',
    region: str = 'USA',
    delay: int = 1,
    universe: str = 'TOP3000',
    dataset_id: str = '',
    search: str = ''
):
    s = global_session
    if not s:
        return {"error": "请先登录获取会话"}
    
    try:
        if len(search) == 0:
            url_template = "https://api.worldquantbrain.com/data-fields?" +\
                f"&instrumentType={instrument_type}" +\
                f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" +\
                "&offset={x}"
            count = s.get(url_template.format(x=0)).json()['count'] 
        else:
            url_template = "https://api.worldquantbrain.com/data-fields?" +\
                f"&instrumentType={instrument_type}" +\
                f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" +\
                f"&search={search}" +\
                "&offset={x}"
            count = 100
        
        datafields_list = []
        for x in range(0, count, 50):
            datafields = s.get(url_template.format(x=x))
            datafields.raise_for_status()
            datafields_list.append(datafields.json()['results'])
        
        datafields_list_flat = [item for sublist in datafields_list for item in sublist]
        datafields_df = pd.DataFrame(datafields_list_flat)
        return {"status": "success", "data": datafields_df.to_dict('records')}
    except requests.exceptions.RequestException as e:
        return {"error": f"获取数据字段失败：{str(e)}"}
    except Exception as e:
        return {"error": f"数据处理失败：{str(e)}"}

# 修正后的工具函数JSON Schema定义（移除Session参数）
tools = [
    {
        "type": "function",
        "function": {
            "name" : "load_credential",
            "description": "获取用户的账号密码信息。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "login",
            "description": "登录用户的worldquant账号",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "邮箱账号，例如：1234567890@gmail.com,12345678@qq.com"
                    },
                    "password":{
                        "type": "string",
                        "description": "用户密码，例如Z%45321"
                    }
                },
                "required": ["username", "password"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datasets",
            "description": "获取某个区域下的所有数据集",
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument_type": {
                        "type": "string",
                        "description": "工具 / 标的类型，默认值'EQUITY'"
                    },
                    "region": {
                        "type": "string",
                        "description": "地区，如'USA','ASI'"
                    },
                    "delay": {
                        "type": "integer",  # JSON Schema用integer而非int
                        "description": "延迟，取值有0,1"
                    },
                    "universe": {
                        "type": "string",
                        "description": "股票池，默认值：'TOP3000'"
                    }
                },
                "required": [],  # 所有参数都有默认值，无需必填
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datafields",
            "description": "获取指定条件下的所有数据字段，支持按数据集ID筛选或关键词搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument_type": {
                        "type": "string",
                        "description": "工具 / 标的类型，默认值'EQUITY'"
                    },
                    "region": {
                        "type": "string",
                        "description": "地区，如'USA','ASI'"
                    },
                    "delay": {
                        "type": "integer",
                        "description": "延迟，取值有0,1"
                    },
                    "universe": {
                        "type": "string",
                        "description": "股票池，默认值：'TOP3000'"
                    },
                    "dataset_id": {
                        "type": "string",
                        "description": "数据集ID，用于筛选指定数据集下的字段，默认值为空字符串"
                    },
                    "search": {
                        "type": "string",
                        "description": "搜索关键词，用于模糊匹配数据字段名称/描述，默认值为空字符串"
                    }
                },
                "required": [],
                "additionalProperties": False
            }
        }
    }
]

# 工具函数映射
available_functions = {
    "load_credential": load_credential,
    "login": login,
    "get_datafields": get_datafields, 
    "get_datasets": get_datasets
}

def run_agent(user_query: str, api_key: str = None, model: str = "qwen-plus"):
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_key or os.getenv("ali_key"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": "你是一个worldquant brain网页的智能助手。你可以登录用户的账号，获取网页中的数据集和数据字段。处理工具调用结果时，要友好地展示给用户。"
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    print("\n" + "="*60)
    print("【用户问题】")
    print(user_query)
    print("="*60)    

    # Agent循环：最多进行5轮工具调用
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")
        
        # 调用大模型
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        tool_calls = response_message.tool_calls

        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("="*60)
            return response_message.content
        
        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")
               
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False, indent=2)}")
     
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                try:
                    # 执行工具函数
                    function_response = function_to_call(**function_args)
                    # 确保返回结果是可序列化的字符串
                    if isinstance(function_response, dict):
                        function_response_str = json.dumps(function_response, ensure_ascii=False)
                    else:
                        function_response_str = str(function_response)
                    
                    print(f"工具返回: {function_response_str[:200]}..." if len(function_response_str) > 200 else f"工具返回: {function_response_str}")
                    
                    # 将工具调用结果加入对话历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_response_str
                    })
                except Exception as e:
                    error_msg = f"执行工具 {function_name} 出错：{str(e)}"
                    print(error_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps({"error": error_msg}, ensure_ascii=False)
                    })
            else:
                error_msg = f"错误：未找到工具 {function_name}"
                print(error_msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps({"error": error_msg}, ensure_ascii=False)
                })
    
    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"

if __name__ == "__main__":
    while True:
        user_query = input("请输入：\n")
        run_agent(user_query)