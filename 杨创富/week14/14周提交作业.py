"""
医疗信息查询助手示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
【已适配阿里百炼DashScope OpenAI兼容模式，可直接运行】
"""
import os
import json
from openai import OpenAI
from openai import AuthenticationError, APIError

# ==================== 关键配置（请替换为自己的阿里百炼API密钥）====================
# 方式1：直接设置（推荐测试用，替换后即可运行）
API_KEY = ""  # 替换为你的百炼API密钥
# 方式2：注释上面一行，通过环境变量读取（生产用）：设置系统环境变量 DASHSCOPE_API_KEY

# ==================== 医疗工具函数定义（核心改造：替换为医疗相关逻辑）====================
def get_disease_list():
    """获取常见疾病分类及基础列表"""
    diseases = [
        {
            "id": "cold_001",
            "name": "普通感冒",
            "type": "呼吸内科",
            "description": "由病毒感染引起的上呼吸道炎症，四季均可发病"
        },
        {
            "id": "fever_001",
            "name": "急性发热",
            "type": "全科/急诊科",
            "description": "体温超过37.3℃的急性全身性症状，多由感染/炎症引起"
        },
        {
            "id": "stomachache_001",
            "name": "急性腹痛",
            "type": "消化内科/普外科",
            "description": "腹部突发的疼痛症状，病因包括肠胃痉挛、肠胃炎、结石等"
        },
        {
            "id": "headache_001",
            "name": "紧张性头痛",
            "type": "神经内科",
            "description": "由精神紧张、疲劳引发的双侧头部紧箍样疼痛"
        },
        {
            "id": "diarrhea_001",
            "name": "急性腹泻",
            "type": "消化内科",
            "description": "每日排便次数增多、粪便稀薄，多由肠道感染、饮食不当引起"
        }
    ]
    return json.dumps(diseases, ensure_ascii=False)

def get_disease_detail(disease_id: str):
    """获取指定疾病的详细信息，参数：disease_id-疾病ID"""
    disease_details = {
        "cold_001": {
            "id": "cold_001",
            "name": "普通感冒",
            "department": "呼吸内科",
            "symptoms": ["鼻塞", "流涕", "咽痛", "轻微咳嗽", "低热（37.3-38℃）"],
            "causes": "鼻病毒、冠状病毒等上呼吸道病毒感染，受凉/劳累易诱发",
            "home_care": ["多喝温水", "保证休息", "清淡饮食", "用生理盐水洗鼻"],
            "see_doctor": ["发热超过38.5℃", "咳嗽加重伴咳痰", "症状持续超过7天", "出现呼吸困难"]
        },
        "fever_001": {
            "id": "fever_001",
            "name": "急性发热",
            "department": "全科/急诊科",
            "symptoms": ["体温升高（≥37.3℃）", "畏寒", "乏力", "肌肉酸痛"],
            "causes": "病毒/细菌感染（感冒、肺炎、肠胃炎等）、炎症反应、免疫性疾病",
            "home_care": ["物理降温（温水擦浴）", "补充水分", "减少衣物", "监测体温"],
            "see_doctor": ["体温≥39℃", "持续发热超过3天", "伴抽搐/呼吸困难/意识模糊"]
        },
        "stomachache_001": {
            "id": "stomachache_001",
            "name": "急性腹痛",
            "department": "消化内科/普外科",
            "symptoms": ["腹部突发疼痛", "恶心", "呕吐", "腹胀"],
            "causes": "肠胃痉挛、急性肠胃炎、泌尿系结石、阑尾炎（转移性右下腹痛）",
            "home_care": ["暂停饮食1-2小时", "轻柔按摩腹部", "热敷腹部（排除结石/阑尾炎）"],
            "see_doctor": ["剧烈腹痛无法缓解", "伴高烧/呕吐血/便血", "腹部僵硬拒按"]
        },
        "headache_001": {
            "id": "headache_001",
            "name": "紧张性头痛",
            "department": "神经内科",
            "symptoms": ["双侧头部紧箍样/压迫性疼痛", "颈部肌肉僵硬", "头晕"],
            "causes": "精神紧张、睡眠不足、过度劳累、颈部肌肉劳损",
            "home_care": ["保证睡眠", "放松心情", "按摩颈部肌肉", "避免强光/噪音刺激"],
            "see_doctor": ["头痛剧烈伴呕吐", "视力模糊", "肢体麻木", "头痛持续超过24小时"]
        },
        "diarrhea_001": {
            "id": "diarrhea_001",
            "name": "急性腹泻",
            "department": "消化内科",
            "symptoms": ["排便次数增多（≥3次/日）", "粪便稀薄/水样", "腹痛", "腹胀"],
            "causes": "肠道病毒/细菌感染、生冷饮食、食物中毒、腹部受凉",
            "home_care": ["口服补液盐防脱水", "清淡饮食（白粥/面条）", "暂停油腻/生冷食物"],
            "see_doctor": ["腹泻伴高烧", "大便带血/黏液", "脱水（口干/尿少/头晕）", "持续超过3天"]
        }
    }
    if disease_id in disease_details:
        return json.dumps(disease_details[disease_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "未查询到该疾病信息"}, ensure_ascii=False)

def get_medication_suggestion(disease_id: str, is_adult: bool = True):
    """获取疾病的用药建议，参数：disease_id-疾病ID，is_adult-是否成人（默认True）"""
    medications = {
        "cold_001": {
            "adult": [
                {"name": "复方氨酚烷胺胶囊", "usage": "1粒/次，2次/日，饭后服"},
                {"name": "连花清瘟胶囊", "usage": "4粒/次，3次/日，饭后服"},
                {"name": "西瓜霜润喉片", "usage": "1片/次，每2小时1次，含服"}
            ],
            "child": [
                {"name": "小儿氨酚黄那敏颗粒", "usage": "1-3岁半袋/次，3次/日，温水冲"},
                {"name": "小儿豉翘清热颗粒", "usage": "按年龄遵医嘱，温水冲服"},
                {"name": "开喉剑喷雾剂", "usage": "每次1-2喷，3-4次/日，喷咽喉"}
            ],
            "note": "用药前需排除过敏，症状轻微可不用药，仅对症护理"
        },
        "fever_001": {
            "adult": [
                {"name": "布洛芬缓释胶囊", "usage": "1粒/次，2次/日，间隔12小时"},
                {"name": "对乙酰氨基酚片", "usage": "1片/次，每4-6小时1次，每日不超过4片"}
            ],
            "child": [
                {"name": "布洛芬混悬液", "usage": "按体重遵医嘱，每6-8小时1次"},
                {"name": "对乙酰氨基酚混悬滴剂", "usage": "按体重遵医嘱，每4-6小时1次"}
            ],
            "note": "体温≥38.5℃再用退烧药，两种退烧药不可交替使用"
        },
        "diarrhea_001": {
            "adult": [
                {"name": "蒙脱石散", "usage": "1袋/次，3次/日，温水冲，饭前服"},
                {"name": "口服补液盐Ⅲ", "usage": "1袋冲500ml温水，随时饮用"},
                {"name": "双歧杆菌三联活菌胶囊", "usage": "2粒/次，3次/日，温水送服"}
            ],
            "child": [
                {"name": "蒙脱石散（儿童装）", "usage": "按年龄遵医嘱，饭前服"},
                {"name": "口服补液盐Ⅲ", "usage": "按体重遵医嘱，少量多次饮用"},
                {"name": "布拉氏酵母菌散", "usage": "1袋/次，1-2次/日，温水冲"}
            ],
            "note": "细菌感染性腹泻需遵医嘱用抗生素，不可自行服用"
        }
    }
    if disease_id not in medications:
        return json.dumps({"error": "该疾病暂无推荐用药信息"}, ensure_ascii=False)
    med_type = "adult" if is_adult else "child"
    result = {
        "disease_id": disease_id,
        "applicable人群": "成人" if is_adult else "儿童",
        "recommended_medications": medications[disease_id][med_type],
        "important_note": medications[disease_id]["note"]
    }
    return json.dumps(result, ensure_ascii=False)

def recommend_department(symptom: str):
    """根据症状推荐就诊科室，参数：symptom-用户描述的症状（如：头痛、腹痛、发烧）"""
    # 症状与科室映射（支持模糊匹配核心关键词）
    symptom_department = {
        "头痛": "神经内科",
        "头晕": "神经内科/耳鼻喉科",
        "发烧": "全科/急诊科",
        "咳嗽": "呼吸内科",
        "鼻塞": "呼吸内科/耳鼻喉科",
        "腹痛": "消化内科/普外科",
        "腹泻": "消化内科",
        "呕吐": "消化内科/急诊科",
        "关节痛": "风湿免疫科/骨科",
        "皮疹": "皮肤科",
        "咽痛": "呼吸内科/耳鼻喉科",
        "胸闷": "心血管内科/呼吸内科"
    }
    # 模糊匹配症状关键词
    matched_dept = None
    for key in symptom_department.keys():
        if key in symptom:
            matched_dept = symptom_department[key]
            break
    if matched_dept:
        result = {
            "symptom": symptom,
            "recommended_department": matched_dept,
            "note": "若症状紧急（如剧烈疼痛/呼吸困难），请直接前往急诊科就诊"
        }
    else:
        result = {
            "symptom": symptom,
            "recommended_department": "全科",
            "note": "症状描述不明确，建议先挂全科号进行初步诊断"
        }
    return json.dumps(result, ensure_ascii=False)

def check_symptom_risk(symptom: str, duration: int, has_other: bool = False):
    """判断症状风险等级，参数：symptom-症状，duration-持续时间（小时），has_other-是否有其他伴随症状（默认False）"""
    # 高危症状关键词
    high_risk_symptoms = ["呼吸困难", "抽搐", "呕血", "便血", "意识模糊", "剧烈疼痛", "高烧不退"]
    # 中危症状关键词
    mid_risk_symptoms = ["持续发热", "反复呕吐", "严重腹泻", "头晕乏力", "胸闷气短"]
    
    risk_level = "低风险"
    suggestion = "可居家观察，对症护理，无需立即就医"
    # 判定风险等级
    if any(key in symptom for key in high_risk_symptoms) or (has_other and duration >= 2):
        risk_level = "高风险"
        suggestion = "⚠️  紧急风险！请立即前往医院急诊科就诊，切勿拖延"
    elif any(key in symptom for key in mid_risk_symptoms) or (duration >= 24) or (has_other and duration >= 12):
        risk_level = "中风险"
        suggestion = "建议在24小时内前往对应科室就诊，避免病情加重"
    
    result = {
        "symptom": symptom,
        "duration_hours": duration,
        "has_other_symptoms": has_other,
        "risk_level": risk_level,
        "medical_suggestion": suggestion
    }
    return json.dumps(result, ensure_ascii=False)

# ==================== 医疗工具函数的JSON Schema定义（适配Function Call）====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_disease_list",
            "description": "获取常见疾病的分类、名称及基础描述列表，无参数",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_disease_detail",
            "description": "获取指定疾病的详细信息，包括症状、病因、居家护理、就医指征",
            "parameters": {
                "type": "object",
                "properties": {"disease_id": {"type": "string", "description": "疾病ID，例如：cold_001, fever_001"}},
                "required": ["disease_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_medication_suggestion",
            "description": "获取指定疾病的针对性用药建议，区分成人和儿童",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease_id": {"type": "string", "description": "疾病ID"},
                    "is_adult": {"type": "boolean", "description": "是否为成人，默认True，儿童请设为False"}
                },
                "required": ["disease_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_department",
            "description": "根据用户描述的症状，推荐对应的医院就诊科室",
            "parameters": {
                "type": "object",
                "properties": {"symptom": {"type": "string", "description": "用户描述的症状，如：头痛、发烧、腹痛"}},
                "required": ["symptom"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_symptom_risk",
            "description": "根据症状、持续时间和伴随症状，判断症状的风险等级并给出就医建议",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom": {"type": "string", "description": "具体症状描述"},
                    "duration": {"type": "integer", "description": "症状持续时间，单位：小时"},
                    "has_other": {"type": "boolean", "description": "是否有其他伴随症状，默认False"}
                },
                "required": ["symptom", "duration"]
            }
        }
    }
]

# ==================== Agent核心逻辑（保留原架构，适配医疗场景）====================
available_functions = {
    "get_disease_list": get_disease_list,
    "get_disease_detail": get_disease_detail,
    "get_medication_suggestion": get_medication_suggestion,
    "recommend_department": recommend_department,
    "check_symptom_risk": check_symptom_risk
}

def run_agent(user_query: str, api_key: str = None, model: str = "qwen-plus"):
    """
    运行医疗信息查询Agent，处理用户医疗咨询（已适配阿里百炼，添加异常捕获）
    Args:
        user_query: 用户输入的医疗问题/症状描述
        api_key: API密钥（优先传参>全局变量>环境变量）
        model: 使用的模型名称
    """
    # 优先级：传参api_key > 全局API_KEY > 系统环境变量
    final_api_key = api_key or API_KEY or os.getenv("DASHSCOPE_API_KEY")
    if not final_api_key:
        print("❌ 错误：未配置API密钥！请在代码中设置API_KEY或配置环境变量DASHSCOPE_API_KEY")
        return "抱歉，未配置有效API密钥，无法处理您的请求。"

    # 初始化OpenAI客户端（阿里百炼兼容模式）
    try:
        client = OpenAI(
            api_key=final_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    except Exception as e:
        print(f"❌ 客户端初始化失败：{str(e)}")
        return "抱歉，客户端初始化失败，请检查API密钥格式。"

    # 初始化对话历史（医疗顾问系统提示词）
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的医疗信息咨询助手，严格按照工具返回结果回答问题，不编造医疗信息。
你可以：1.查询常见疾病列表及详细信息 2.根据疾病给出成人/儿童用药建议 3.根据症状推荐就诊科室 4.判断症状风险等级并给出就医建议。
注意：本助手仅提供基础医疗信息参考，不构成专业诊疗建议，重症/急症请立即前往医院就诊！
工具返回的json数据请格式化后清晰展示，给用户易懂的解答。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    print("\n" + "="*80)
    print(f"【用户医疗咨询】：{user_query}")
    print("="*80)

    # Agent循环：最多5轮工具调用（处理复杂查询）
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考（工具调用循环）---")

        # 调用大模型（temperature=0.1保证回答稳定性，避免医疗信息偏差）
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1
            )
        except AuthenticationError:
            print("❌ 错误：API密钥无效！请检查密钥是否正确，是否有百炼平台调用权限。")
            return "抱歉，API密钥无效，请检查后重试。"
        except APIError as e:
            print(f"❌ 模型调用失败：{str(e)}")
            return f"抱歉，模型服务调用失败：{str(e)}"
        except Exception as e:
            print(f"❌ 未知错误：{str(e)}")
            return f"抱歉，处理请求时发生未知错误：{str(e)}"

        response_message = response.choices[0].message
        messages.append(response_message)  # 将模型响应加入对话历史

        # 检查是否需要调用工具：无工具调用则返回最终答案
        tool_calls = response_message.tool_calls
        if not tool_calls:
            print("\n【医疗助手最终回复】：")
            print(response_message.content)
            print("="*80)
            return response_message.content

        # 执行工具调用：批量处理模型指定的工具
        print(f"\n【Agent决定调用 {len(tool_calls)} 个医疗工具】")
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n🔧 医疗工具名称：{function_name}")
            print(f"📌 工具调用参数：{json.dumps(function_args, ensure_ascii=False, indent=2)}")

            # 执行对应的医疗工具函数
            try:
                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)
                    # 截断过长的返回结果（调试友好，不影响实际逻辑）
                    show_response = function_response if len(function_response) <= 500 else f"{function_response[:500]}..."
                    print(f"✅ 工具返回结果：{show_response}")

                    # 百炼兼容模式：将工具返回结果加入对话历史（role=tool）
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_response
                    })
                else:
                    # 工具不存在的异常处理
                    error_msg = json.dumps({"error": f"医疗工具 {function_name} 未定义"}, ensure_ascii=False)
                    print(f"❌ 错误：未找到指定医疗工具 {function_name}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": error_msg
                    })
            except Exception as e:
                # 工具调用执行失败的异常处理
                error_msg = json.dumps({"error": f"医疗工具调用失败：{str(e)}"}, ensure_ascii=False)
                print(f"❌ 工具调用执行失败：{str(e)}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": error_msg
                })

    # 达到最大工具调用次数，终止处理
    print("\n⚠️  警告：达到最大工具调用次数（5轮），结束当前医疗咨询处理")
    print("="*80)
    return "抱歉，您的医疗咨询问题较复杂，超出当前处理能力，请简化问题后重新咨询（注：重症请立即就医）。"

# ==================== 示例场景（直接运行，无需额外配置）====================
if __name__ == "__main__":
    # 示例1：查询常见疾病列表（基础查询）
    run_agent("你们能查询哪些常见疾病？", model="qwen-plus")

    # 示例2：查询具体疾病详情（单工具调用）
    # run_agent("我想了解普通感冒的详细信息，包括症状和护理方法", model="qwen-plus")

    # 示例3：获取儿童用药建议（带参数工具调用）
    # run_agent("4岁孩子感冒了，有什么合适的用药建议？", model="qwen-plus")

    # 示例4：根据症状推荐科室（模糊匹配）
    # run_agent("我一直头痛，应该挂哪个科的号？", model="qwen-plus")

    # 示例5：判断症状风险等级（多参数）
    # run_agent("我腹痛了8小时，还伴有呕吐，要不要立刻去医院？", model="qwen-plus")

    # 自定义医疗咨询
    # run_agent("我发烧39℃，持续了6小时，还有呼吸困难，该怎么办？", model="qwen-plus")