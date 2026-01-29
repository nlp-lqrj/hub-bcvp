# -*- coding:utf-8 -*-

"""
所有可用工具（函数）
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_all_products",
            "description": "获取所有可用的车险产品列表，包括产品名称、类型、产品描述",
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
            "name": "get_product_detail",
            "description": "获取指定车险产品的详细信息，包括保障范围和免责条款等等",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品ID，例如：compulsory_insurance"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_premium",
            "description": "根据车辆信息、驾驶人信息和选择的险种计算总保费",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要投保的产品ID列表"
                    },
                    "car_price": {
                        "type": "integer",
                        "description": "车辆购置价格"
                    },
                    "car_age": {
                        "type": "integer",
                        "description": "车辆年龄"
                    },
                    "driver_age": {
                        "type": "integer",
                        "description": "驾驶员年龄"
                    }
                },
                "required": ["product_ids", "car_price", "car_age", "driver_age"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": "检查车辆和驾驶人是否符合基本投保资格",
            "parameters": {
                "type": "object",
                "properties": {
                    "car_age": {
                        "type": "integer",
                        "description": "车辆年龄"
                    },
                    "driver_license_years": {
                        "type": "integer",
                        "description": "驾龄"
                    }
                },
                "required": ["vehicle_age", "driver_license_years"]
            }
        }
    }
]