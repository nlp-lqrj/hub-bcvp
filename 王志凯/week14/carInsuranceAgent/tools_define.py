# -*- coding:utf-8 -*-

"""
工具集：
1.获取所有车险产品
2.每个车险品种的详细介绍
3.判断用户是否具有投保资格
4.根据用户信息和需求计算保费
"""

import json

# 获取所有车险产品
def get_all_products():
    products = [
        {
            "id": "compulsory_insurance",
            "name": "机动车交通事故责任强制保险",
            "type": "强制保险",
            "description": "法律规定必须投保的险种，保障第三方人身伤亡和财产损失"
        },
        {
            "id": "damage_insurance",
            "name": "机动车损失保险",
            "type": "商业保险",
            "description": "保障被保险车辆因意外事故或自然灾害造成的车辆损失"
        },
        {
            "id": "liability_insurance",
            "name": "机动车第三者责任保险",
            "type": "商业保险",
            "description": "保障因车辆使用导致第三方遭受人身伤亡或财产损失的赔偿责任"
        }
    ]

    # 序列化：将python对象转化为JSON字符串（需要放入content中）
    # ensure_ascii: JSON 模块为了保证兼容性，默认会把中文转换成 Unicode 编码，最好设置为False
    return json.dumps(products, ensure_ascii=False)


# 根据product_id获取指定车险产品的详细信息
def get_product_detail(product_id: str):
    products = {
        "compulsory_insurance": {
            "id": "compulsory_insurance",
            "name": "机动车交通事故责任强制保险",
            "type": "强制保险",
            "description": "法律规定必须投保的险种",
            "coverage_items": ["第三方人身伤亡", "第三方财产损失"],
            "note": "费率根据车辆座位数和出险记录浮动"
        },
        "damage_insurance": {
            "id": "damage_insurance",
            "name": "机动车损失保险",
            "type": "商业保险",
            "description": "保障被保险车辆自身损失",
            "coverage_items": ["碰撞", "火灾", "自然灾害", "外界物体坠落"],
            "exclusions": ["自然磨损", "轮胎单独损坏", "战争"],
            "note": "包含不计免赔服务"
        },
        "liability_insurance": {
            "id": "liability_insurance",
            "name": "机动车第三者责任保险",
            "type": "商业保险",
            "description": "保障第三方高额赔偿",
            "coverage_items": ["第三方医疗费用", "第三方车辆维修", "死亡伤残赔偿"],
            "limits": ["50万", "100万", "200万", "300万"]
        }
    }

    if product_id in products:
        return json.dumps(products[product_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)


# 根据用户信息和需求计算保费
def calculate_premium(product_ids: list, car_price: int, car_age: int, driver_age: int):
    """
    参数：
    product_ids: 选购的产品ID列表
    car_price:   车辆购置价
    car_age:     车龄
    driver_age:  驾驶员年龄
    """

    # 基础费率配置
    rates = {
        "compulsory_insurance": 0.01,  # 交强险按车价比例
        "damage_insurance": 0.008,     # 车损险按车价比例
        "liability_insurance": 1200    # 三者险按固定年费
    }

    # 驾龄打折系数：驾龄越长越便宜，最多打7折
    driver_age_ratio = max(0.7, 1.2 - (driver_age * 0.02))
    # 车龄打折系数：车龄越长越便宜，最多打8折
    car_age_ratio = max(0.8, 1 - (car_age * 0.03))

    # 总保费
    total = 0
    # 每种车险的保费
    insurance_cost = {}

    for pid in product_ids:
        if pid == "compulsory_insurance":
            cost = car_price * rates[pid]
            insurance_cost["交强险"] = round(cost, 2)
            total += cost
        elif pid == "damage_insurance":
            cost = car_price * rates[pid] * car_age_ratio * driver_age_ratio
            insurance_cost["车损险"] = round(cost, 2)
            total += cost
        elif pid == "liability_insurance":
            cost = rates[pid] * driver_age_ratio
            insurance_cost["三者险"] = round(cost, 2)
            total += cost

    result = {
        "选购的车险产品": product_ids,
        "车辆购置价格": car_price,
        "驾驶员年龄": driver_age,
        "每种车险的保费": insurance_cost,
        "总保费": round(total, 2),
        "单位": "元"
    }

    return json.dumps(result, ensure_ascii=False)


# 检查用户是否有参保资格
def check_eligibility(car_age: int, driver_license_years: int):
    """
    参数：
    car_age: 车龄
    driver_license_years: 驾龄
    """
    if car_age > 10:
        return json.dumps({"eligible": False, "reason": "车辆年限超过10年，不支持投保"}, ensure_ascii=False)

    if driver_license_years < 1:
        return json.dumps({"eligible": False, "reason": "驾龄不足1年，暂不支持投保"}, ensure_ascii=False)

    return json.dumps({"eligible": True, "message": "符合投保条件"}, ensure_ascii=False)


