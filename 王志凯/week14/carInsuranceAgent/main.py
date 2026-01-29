# -*- coding:utf-8 -*-

"""
医疗Agent：演示 Function Call 功能
用户问题 - 工具调用 - 整合回答 - 打印LLM最终回复结果
"""

import os
from tools import tools
from tools_define import *
from openai import OpenAI

# 工具名称-函数 映射
function_mapping = {
    "get_all_products": get_all_products,
    "get_product_detail": get_product_detail,
    "calculate_premium": calculate_premium,
    "check_eligibility": check_eligibility
}

def run_agent(query: str, model: str = "qwen_plus"):
    client = OpenAI(
        # API_KEY设置到了环境变量中，不需要显示入参
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    messages = [
        {
            "role": "system",
            "content": """
                        假设你是一个资深的车险顾问，你能完成：
                        1.获取所有车险产品
                        2.每个车险品种的详细介绍
                        3.判断用户是否具有投保资格
                        4.根据用户信息和需求计算保费
                        等功能。
                      """
        },
        {
            "role": "user",
            "content": query
        }
    ]

    # 为避免死循环，或重复调用多次，设置最多调用5次
    for i in range(5):
        print(f"====第{i + 1}轮工具调用====")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"

        )
        message = response.choices[0].message
        messages.append(message)

        content = message.content
        tool_calls = message.tool_calls

        if tool_calls:
            print(f"大模型判断需要调用的工具：{[tool.function.name for tool in tool_calls]}")
            for tool in tool_calls:
                # 1.从返回的tool_calls中获取函数信息
                function_id = tool.id
                function_name = tool.function.name
                function_args = json.loads(tool.function.arguments)

                if function_name in function_mapping:
                    # 2.执行函数
                    function_instance = function_mapping[function_name]
                    result = function_instance(**function_args)
                    print(f"工具 {function_name} 返回的结果为：{result}")

                    # 3.将返回结果封装到原始messages中，准备下一次API调用
                    result_message = {
                        "role": "tool",
                        "tool_call_id": function_id,
                        "name": function_name,
                        "content": result
                    }
                    messages.append(result_message)
                else:
                    print(f"未找到工具{function_name}，请检查该工具是否注册")
                    return
        else:
            # 如果tool_calls为空，则content不为空，说明模型直接生成了回复
            print(f"【Agent最终回复】：{content}")
            break


if __name__ == "__main__":
    examples = [
        "你们目前都有哪些车险产品？",
        "车损险是干嘛的？具体保什么？",
        "我想对比一下两种方案：方案一买交强险和车损险；方案二只买三者险和交强险。我今年25岁，车价10万，车龄2年。",
        "我的车已经用了12年了，我开了5年车，想给我的车买保险，怎么买比较合适呢？"
    ]

    run_agent(examples[3], model="qwen-plus")


    # client = OpenAI(
    #     api_key=os.getenv("DASHSCOPE_API_KEY"),
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    # completion = client.chat.completions.create(
    #     model="qwen-plus",
    #     messages=[{'role': 'user', 'content': '你是谁？'}]
    # )
    # print(completion.choices[0].message.content)