"""
家用汽车推荐系统 - 演示大语言模型的Function Call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from openai import OpenAI


# ==================== 工具函数定义 ====================
# 汽车推荐系统的核心功能实现

def get_car_models():
    """
    获取汽车品牌、型号列表
    包含进口车、合资车、国产车的小轿车和SUV各一辆
    """
    cars = [
        {
            "id": "bmw_320i",
            "brand": "宝马",
            "model": "320i",
            "type": "小轿车",
            "origin": "进口车",
            "price_range": "28-35万",
            "fuel_type": "汽油",
            "transmission": "自动"
        },
        {
            "id": "toyota_camry",
            "brand": "丰田",
            "model": "凯美瑞",
            "type": "小轿车",
            "origin": "合资车",
            "price_range": "18-25万",
            "fuel_type": "汽油/混动",
            "transmission": "自动"
        },
        {
            "id": "geely_emgrand",
            "brand": "吉利",
            "model": "帝豪",
            "type": "小轿车",
            "origin": "国产车",
            "price_range": "6-10万",
            "fuel_type": "汽油",
            "transmission": "手动/自动"
        },
        {
            "id": "audi_q5",
            "brand": "奥迪",
            "model": "Q5L",
            "type": "SUV",
            "origin": "进口车",
            "price_range": "35-45万",
            "fuel_type": "汽油/混动",
            "transmission": "自动"
        },
        {
            "id": "honda_crv",
            "brand": "本田",
            "model": "CR-V",
            "type": "SUV",
            "origin": "合资车",
            "price_range": "16-24万",
            "fuel_type": "汽油/混动",
            "transmission": "自动"
        },
        {
            "id": "chery_tiggo",
            "brand": "奇瑞",
            "model": "瑞虎8",
            "type": "SUV",
            "origin": "国产车",
            "price_range": "9-15万",
            "fuel_type": "汽油",
            "transmission": "手动/自动"
        }
    ]
    return json.dumps(cars, ensure_ascii=False)


def get_car_detail(car_id: str):
    """
    获取指定汽车型号的详情信息

    Args:
        car_id: 汽车ID
    """
    car_details = {
        "bmw_320i": {
            "id": "bmw_320i",
            "brand": "宝马",
            "model": "320i",
            "type": "小轿车",
            "origin": "进口车",
            "base_price": 289000,
            "engine": "2.0T涡轮增压",
            "horsepower": 184,
            "torque": 300,
            "top_speed": 235,
            "0_100": 7.9,
            "fuel_consumption": 6.2,
            "length": 4719,
            "width": 1827,
            "height": 1459,
            "wheelbase": 2851,
            "seats": 5,
            "standard_features": ["LED大灯", "自动空调", "多功能方向盘", "倒车影像", "定速巡航"],
            "optional_features": [
                {"id": "leather_seats", "name": "真皮座椅", "price": 8000},
                {"id": "navigation", "name": "导航系统", "price": 5000},
                {"id": "sunroof", "name": "全景天窗", "price": 10000},
                {"id": "harman_kardon", "name": "哈曼卡顿音响", "price": 12000},
                {"id": "adaptive_cruise", "name": "自适应巡航", "price": 15000}
            ]
        },
        "toyota_camry": {
            "id": "toyota_camry",
            "brand": "丰田",
            "model": "凯美瑞",
            "type": "小轿车",
            "origin": "合资车",
            "base_price": 179800,
            "engine": "2.0L自然吸气/2.5L混动",
            "horsepower": 178,
            "torque": 210,
            "top_speed": 205,
            "0_100": 9.1,
            "fuel_consumption": 5.8,
            "length": 4885,
            "width": 1840,
            "height": 1455,
            "wheelbase": 2825,
            "seats": 5,
            "standard_features": ["LED大灯", "自动空调", "多功能方向盘", "倒车影像", "车道偏离预警"],
            "optional_features": [
                {"id": "leather_seats", "name": "真皮座椅", "price": 6000},
                {"id": "navigation", "name": "导航系统", "price": 4000},
                {"id": "sunroof", "name": "全景天窗", "price": 8000},
                {"id": "jbl_audio", "name": "JBL音响", "price": 9000},
                {"id": "360_camera", "name": "360全景影像", "price": 7000}
            ]
        },
        "geely_emgrand": {
            "id": "geely_emgrand",
            "brand": "吉利",
            "model": "帝豪",
            "type": "小轿车",
            "origin": "国产车",
            "base_price": 62800,
            "engine": "1.5L自然吸气",
            "horsepower": 114,
            "torque": 147,
            "top_speed": 175,
            "0_100": 12.8,
            "fuel_consumption": 5.7,
            "length": 4632,
            "width": 1825,
            "height": 1489,
            "wheelbase": 2650,
            "seats": 5,
            "standard_features": ["卤素大灯", "手动空调", "多功能方向盘", "倒车雷达", "蓝牙连接"],
            "optional_features": [
                {"id": "led_lights", "name": "LED大灯", "price": 2000},
                {"id": "automatic_aircon", "name": "自动空调", "price": 1500},
                {"id": "touchscreen", "name": "中控触摸屏", "price": 3000},
                {"id": "rear_camera", "name": "倒车影像", "price": 1200},
                {"id": "alloy_wheels", "name": "铝合金轮毂", "price": 2500}
            ]
        },
        "audi_q5": {
            "id": "audi_q5",
            "brand": "奥迪",
            "model": "Q5L",
            "type": "SUV",
            "origin": "进口车",
            "base_price": 358800,
            "engine": "2.0T涡轮增压",
            "horsepower": 252,
            "torque": 370,
            "top_speed": 230,
            "0_100": 6.7,
            "fuel_consumption": 7.2,
            "length": 4770,
            "width": 1893,
            "height": 1667,
            "wheelbase": 2908,
            "seats": 5,
            "standard_features": ["LED矩阵大灯", "三区自动空调", "多功能方向盘", "360全景影像", "自适应巡航"],
            "optional_features": [
                {"id": "virtual_cockpit", "name": "虚拟驾驶舱", "price": 12000},
                {"id": "bang_olufsen", "name": "Bang & Olufsen音响", "price": 18000},
                {"id": "head_up_display", "name": "抬头显示", "price": 10000},
                {"id": "air_suspension", "name": "空气悬架", "price": 25000},
                {"id": "driver_assistance", "name": "驾驶辅助包", "price": 20000}
            ]
        },
        "honda_crv": {
            "id": "honda_crv",
            "brand": "本田",
            "model": "CR-V",
            "type": "SUV",
            "origin": "合资车",
            "base_price": 169800,
            "engine": "1.5T涡轮增压/2.0L混动",
            "horsepower": 193,
            "torque": 243,
            "top_speed": 198,
            "0_100": 8.4,
            "fuel_consumption": 6.4,
            "length": 4621,
            "width": 1855,
            "height": 1679,
            "wheelbase": 2661,
            "seats": 5,
            "standard_features": ["LED大灯", "双区自动空调", "多功能方向盘", "倒车影像", "车道保持"],
            "optional_features": [
                {"id": "leather_seats", "name": "真皮座椅", "price": 6000},
                {"id": "panoramic_sunroof", "name": "全景天窗", "price": 8000},
                {"id": "wireless_charging", "name": "无线充电", "price": 2000},
                {"id": "blind_spot", "name": "盲点监测", "price": 4000},
                {"id": "hands_free_tailgate", "name": "电动尾门", "price": 5000}
            ]
        },
        "chery_tiggo": {
            "id": "chery_tiggo",
            "brand": "奇瑞",
            "model": "瑞虎8",
            "type": "SUV",
            "origin": "国产车",
            "base_price": 88800,
            "engine": "1.6T涡轮增压",
            "horsepower": 197,
            "torque": 290,
            "top_speed": 190,
            "0_100": 9.5,
            "fuel_consumption": 6.8,
            "length": 4700,
            "width": 1860,
            "height": 1746,
            "wheelbase": 2710,
            "seats": 5,
            "standard_features": ["LED大灯", "手动空调", "多功能方向盘", "倒车雷达", "蓝牙连接"],
            "optional_features": [
                {"id": "automatic_aircon", "name": "自动空调", "price": 2000},
                {"id": "sunroof", "name": "天窗", "price": 4000},
                {"id": "touchscreen", "name": "中控触摸屏", "price": 3500},
                {"id": "360_camera", "name": "360全景影像", "price": 4500},
                {"id": "electric_seats", "name": "电动座椅", "price": 3000}
            ]
        }
    }

    if car_id in car_details:
        return json.dumps(car_details[car_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "汽车型号不存在"}, ensure_ascii=False)


def calculate_car_price(car_id: str, options: list):
    """
    计算指定品牌型号的汽车进行选配后的购买价格

    Args:
        car_id: 汽车ID
        options: 选配ID列表
    """
    # 首先获取汽车基础信息
    car_detail = json.loads(get_car_detail(car_id))

    if "error" in car_detail:
        return json.dumps({"error": "汽车型号不存在"}, ensure_ascii=False)

    base_price = car_detail["base_price"]
    optional_features = car_detail["optional_features"]

    # 计算选配价格
    options_price = 0
    selected_options = []

    for option_id in options:
        for feature in optional_features:
            if feature["id"] == option_id:
                options_price += feature["price"]
                selected_options.append(feature["name"])
                break

    # 计算购置税（约为裸车价的8.547%）
    tax = (base_price + options_price) * 0.08547

    # 计算保险费用（约为裸车价的3-5%）
    insurance = (base_price + options_price) * 0.04

    # 计算上牌费用
    registration_fee = 500

    # 总落地价
    total_price = base_price + options_price + tax + insurance + registration_fee

    result = {
        "car_id": car_id,
        "brand": car_detail["brand"],
        "model": car_detail["model"],
        "base_price": base_price,
        "options": selected_options,
        "options_price": options_price,
        "tax": round(tax, 2),
        "insurance": round(insurance, 2),
        "registration_fee": registration_fee,
        "total_price": round(total_price, 2),
        "price_breakdown": f"裸车价: {base_price}元, 选配: {options_price}元, 购置税: {round(tax, 2)}元, 保险: {round(insurance, 2)}元, 上牌: {registration_fee}元"
    }

    return json.dumps(result, ensure_ascii=False)


def calculate_car_cost(car_id: str, annual_mileage: int):
    """
    计算指定品牌型号汽车驾驶的油耗和用车成本

    Args:
        car_id: 汽车ID
        annual_mileage: 年行驶里程（公里）
    """
    # 获取汽车基础信息
    car_detail = json.loads(get_car_detail(car_id))

    if "error" in car_detail:
        return json.dumps({"error": "汽车型号不存在"}, ensure_ascii=False)

    # 基础参数
    fuel_consumption = car_detail["fuel_consumption"]  # 百公里油耗
    fuel_price = 7.5  # 汽油价格（元/升）

    # 计算燃油成本
    annual_fuel_cost = (annual_mileage / 100) * fuel_consumption * fuel_price

    # 计算保养成本
    if car_detail["origin"] == "进口车":
        maintenance_cost = 8000  # 进口车保养较贵
    elif car_detail["origin"] == "合资车":
        maintenance_cost = 5000  # 合资车保养适中
    else:
        maintenance_cost = 3000  # 国产车保养较便宜

    # 计算保险费用（每年）
    insurance_cost = car_detail["base_price"] * 0.03

    # 计算年检费用
    inspection_fee = 300

    # 计算停车费用
    parking_fee = 3000

    # 计算其他费用（洗车、过路费等）
    other_costs = 2000

    # 总年度用车成本
    total_annual_cost = annual_fuel_cost + maintenance_cost + insurance_cost + inspection_fee + parking_fee + other_costs

    # 计算每公里成本
    cost_per_km = total_annual_cost / annual_mileage

    result = {
        "car_id": car_id,
        "brand": car_detail["brand"],
        "model": car_detail["model"],
        "annual_mileage": annual_mileage,
        "fuel_consumption": fuel_consumption,
        "fuel_cost": round(annual_fuel_cost, 2),
        "maintenance_cost": maintenance_cost,
        "insurance_cost": round(insurance_cost, 2),
        "inspection_fee": inspection_fee,
        "parking_fee": parking_fee,
        "other_costs": other_costs,
        "total_annual_cost": round(total_annual_cost, 2),
        "cost_per_km": round(cost_per_km, 2),
        "cost_breakdown": f"燃油: {round(annual_fuel_cost, 2)}元, 保养: {maintenance_cost}元, 保险: {round(insurance_cost, 2)}元, 年检: {inspection_fee}元, 停车: {parking_fee}元, 其他: {other_costs}元"
    }

    return json.dumps(result, ensure_ascii=False)


def compare_cars(car_ids: list, annual_mileage: int):
    """
    比较多个品牌型号的汽车购买价格和后续的用车成本费用

    Args:
        car_ids: 汽车ID列表
        annual_mileage: 年行驶里程（公里）
    """
    comparisons = []

    for car_id in car_ids:
        # 获取汽车详情
        car_detail = json.loads(get_car_detail(car_id))
        if "error" in car_detail:
            continue

        # 计算基础购买价格（无选配）
        base_price = car_detail["base_price"]

        # 计算基础落地价
        tax = base_price * 0.08547
        insurance = base_price * 0.04
        registration_fee = 500
        total_price = base_price + tax + insurance + registration_fee

        # 计算用车成本
        cost_result = json.loads(calculate_car_cost(car_id, annual_mileage))
        if "error" not in cost_result:
            car_comparison = {
                "car_id": car_id,
                "brand": car_detail["brand"],
                "model": car_detail["model"],
                "type": car_detail["type"],
                "origin": car_detail["origin"],
                "base_price": base_price,
                "total_price": round(total_price, 2),
                "annual_cost": cost_result["total_annual_cost"],
                "fuel_consumption": car_detail["fuel_consumption"],
                "cost_per_km": cost_result["cost_per_km"]
            }
            comparisons.append(car_comparison)

    # 按总落地价排序
    comparisons.sort(key=lambda x: x["total_price"])

    result = {
        "comparison_params": {
            "annual_mileage": annual_mileage,
            "car_count": len(comparisons)
        },
        "cars": comparisons
    }

    return json.dumps(result, ensure_ascii=False)


# ==================== 工具函数的JSON Schema定义 ====================
# 这部分会成为LLM的提示词的一部分

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_car_models",
            "description": "获取汽车品牌、型号列表，包含进口车、合资车、国产车的小轿车和SUV各一辆，以及基本价格范围、燃油类型等信息",
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
            "name": "get_car_detail",
            "description": "获取指定汽车型号的详细信息，包括基础价格、发动机参数、尺寸、配置选项等完整信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "car_id": {
                        "type": "string",
                        "description": "汽车ID，例如：bmw_320i, toyota_camry, geely_emgrand, audi_q5, honda_crv, chery_tiggo"
                    }
                },
                "required": ["car_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_car_price",
            "description": "计算指定品牌型号的汽车进行选配后的购买价格，包括裸车价、选配价格、购置税、保险和上牌费用",
            "parameters": {
                "type": "object",
                "properties": {
                    "car_id": {
                        "type": "string",
                        "description": "汽车ID"
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "选配ID列表，可从get_car_detail获取的optional_features中选择"
                    }
                },
                "required": ["car_id", "options"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_car_cost",
            "description": "计算指定品牌型号汽车驾驶的油耗和用车成本，包括燃油费用、保养费用、保险费用等年度总费用",
            "parameters": {
                "type": "object",
                "properties": {
                    "car_id": {
                        "type": "string",
                        "description": "汽车ID"
                    },
                    "annual_mileage": {
                        "type": "integer",
                        "description": "年行驶里程（公里），一般家用车每年约10000-20000公里"
                    }
                },
                "required": ["car_id", "annual_mileage"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cars",
            "description": "比较多个品牌型号的汽车购买价格和后续的用车成本费用，包括落地价、年度用车成本、百公里油耗等指标",
            "parameters": {
                "type": "object",
                "properties": {
                    "car_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要比较的汽车ID列表"
                    },
                    "annual_mileage": {
                        "type": "integer",
                        "description": "年行驶里程（公里）"
                    }
                },
                "required": ["car_ids", "annual_mileage"]
            }
        }
    }
]

# ==================== Agent核心逻辑 ====================

# 工具函数映射
available_functions = {
    "get_car_models": get_car_models,
    "get_car_detail": get_car_detail,
    "calculate_car_price": calculate_car_price,
    "calculate_car_cost": calculate_car_cost,
    "compare_cars": compare_cars
}


def run_agent(user_query: str, api_key: str = None, model: str = "qwen-plus"):
    """
    运行Agent，处理用户查询

    Args:
        user_query: 用户输入的问题
        api_key: API密钥（如果不提供则从环境变量读取）
        model: 使用的模型名称
    """
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的汽车购买顾问助手。你可以：
1. 介绍各种汽车品牌和型号
2. 提供汽车详细配置信息
3. 根据用户需求计算购车价格
4. 分析汽车的使用成本
5. 比较不同汽车的性价比

请根据用户的问题，使用合适的工具来获取信息并给出专业的建议。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    print("\n" + "=" * 60)
    print("【用户问题】")
    print(user_query)
    print("=" * 60)

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
            tool_choice="auto"  # 让模型自主决定是否调用工具
        )

        response_message = response.choices[0].message

        # 将模型响应加入对话历史
        messages.append(response_message)

        # 检查是否需要调用工具
        tool_calls = response_message.tool_calls

        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("=" * 60)
            return response_message.content

        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")

            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)

                print(f"工具返回: {function_response[:200]}..." if len(
                    function_response) > 200 else f"工具返回: {function_response}")

                # 将工具调用结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            else:
                print(f"错误：未找到工具 {function_name}")

    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"


# ==================== 示例场景 ====================

def demo_scenarios():
    """
    演示几个典型场景
    """
    print("\n" + "#" * 60)
    print("# 家用汽车推荐系统演示 - Function Call能力展示")
    print("#" * 60)

    # 注意：需要设置环境变量 DASHSCOPE_API_KEY
    # 或者在调用时传入api_key参数

    scenarios = [
        "你们有哪些汽车型号？",
        "我想了解一下宝马320i的详细信息",
        "我想买丰田凯美瑞，选装配真皮座椅和导航系统，落地价大概多少？",
        "我每年开车15000公里，奥迪Q5L的用车成本大概多少？",
        "帮我比较一下吉利帝豪、丰田凯美瑞和宝马320i，每年开12000公里的情况"
    ]

    print("\n以下是几个示例场景，您可以选择其中一个运行：\n")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")

    print("\n" + "-" * 60)
    print("要运行示例，请取消注释main函数中的相应代码")
    print("并确保设置了环境变量：DASHSCOPE_API_KEY")
    print("-" * 60)


if __name__ == "__main__":
# 展示示例场景
# demo_scenarios()

# 运行示例（取消注释下面的代码来运行）
# 注意：需要先设置环境变量 DASHSCOPE_API_KEY

# 示例1：查询汽车型号列表
# run_agent("你们有哪些汽车型号？", model="qwen-plus")

# 示例2：查询汽车详情
# run_agent("我想了解一下宝马320i的详细信息", model="qwen-plus")

# 示例3：计算购车价格
# run_agent("我想买丰田凯美瑞，选装配真皮座椅和导航系统，落地价大概多少？", model="qwen-plus")

# 示例4：计算用车成本
# run_agent("我每年开车15000公里，奥迪Q5L的用车成本大概多少？", model="qwen-plus")

# 示例5：比较多个汽车
# run_agent("帮我比较一下吉利帝豪、丰田凯美瑞和宝马320i，每年开12000公里的情况", model="qwen-plus")

# 自定义查询
# run_agent("有星越L这款车吗，我想知道它的详细信息", model="qwen-plus")
