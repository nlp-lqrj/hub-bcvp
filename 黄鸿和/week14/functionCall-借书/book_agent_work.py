"""
function call-图书借书
"""

import os
import json
from openai import OpenAI


# ==================== 工具函数定义 ====================

def search_books(keyword: str = None, author: str = None, category: str = None):
    """
    搜索图书
    
    Args:
        keyword: 关键词（书名）
        author: 作者
        category: 分类（如：小说、科技、历史、文学）
    """
    # 模拟图书数据库
    books = [
        {
            "id": "book_001",
            "title": "Python编程从入门到实践",
            "author": "Eric Matthes",
            "category": "科技",
            "isbn": "9787115428028",
            "available": True,
            "total_copies": 5,
            "available_copies": 3
        },
        {
            "id": "book_002",
            "title": "三体",
            "author": "刘慈欣",
            "category": "小说",
            "isbn": "9787536692930",
            "available": True,
            "total_copies": 10,
            "available_copies": 2
        },
        {
            "id": "book_003",
            "title": "活着",
            "author": "余华",
            "category": "文学",
            "isbn": "9787506365437",
            "available": True,
            "total_copies": 8,
            "available_copies": 5
        },
        {
            "id": "book_004",
            "title": "人类简史",
            "author": "尤瓦尔·赫拉利",
            "category": "历史",
            "isbn": "9787508647357",
            "available": False,
            "total_copies": 6,
            "available_copies": 0
        },
        {
            "id": "book_005",
            "title": "机器学习实战",
            "author": "Peter Harrington",
            "category": "科技",
            "isbn": "9787115317957",
            "available": True,
            "total_copies": 4,
            "available_copies": 1
        }
    ]
    
    # 筛选逻辑
    filtered_books = []
    for book in books:
        # 关键词筛选
        if keyword and keyword not in book["title"]:
            continue
        
        # 作者筛选
        if author and author not in book["author"]:
            continue
        
        # 分类筛选
        if category and book["category"] != category:
            continue
        
        filtered_books.append(book)
    
    result = {
        "total": len(filtered_books),
        "books": filtered_books
    }
    
    return json.dumps(result, ensure_ascii=False)


def get_book_detail(book_id: str):
    """
    获取图书详细信息
    
    Args:
        book_id: 图书ID
    """
    books_detail = {
        "book_001": {
            "id": "book_001",
            "title": "Python编程从入门到实践",
            "author": "Eric Matthes",
            "category": "科技",
            "isbn": "9787115428028",
            "publisher": "人民邮电出版社",
            "publish_date": "2016-07-01",
            "pages": 459,
            "price": 89.00,
            "description": "一本针对所有层次Python读者而作的Python入门书",
            "available": True,
            "total_copies": 5,
            "available_copies": 3,
            "location": "A区3层科技类书架"
        },
        "book_002": {
            "id": "book_002",
            "title": "三体",
            "author": "刘慈欣",
            "category": "小说",
            "isbn": "9787536692930",
            "publisher": "重庆出版社",
            "publish_date": "2008-01-01",
            "pages": 302,
            "price": 23.00,
            "description": "科幻小说，雨果奖获奖作品",
            "available": True,
            "total_copies": 10,
            "available_copies": 2,
            "location": "B区2层小说类书架"
        },
        "book_003": {
            "id": "book_003",
            "title": "活着",
            "author": "余华",
            "category": "文学",
            "isbn": "9787506365437",
            "publisher": "作家出版社",
            "publish_date": "2012-08-01",
            "pages": 191,
            "price": 20.00,
            "description": "当代文学经典作品",
            "available": True,
            "total_copies": 8,
            "available_copies": 5,
            "location": "B区1层文学类书架"
        },
        "book_004": {
            "id": "book_004",
            "title": "人类简史",
            "author": "尤瓦尔·赫拉利",
            "category": "历史",
            "isbn": "9787508647357",
            "publisher": "中信出版社",
            "publish_date": "2014-11-01",
            "pages": 440,
            "price": 68.00,
            "description": "从认知革命、农业革命到科学革命，我们真的了解自己吗？",
            "available": False,
            "total_copies": 6,
            "available_copies": 0,
            "location": "C区2层历史类书架"
        },
        "book_005": {
            "id": "book_005",
            "title": "机器学习实战",
            "author": "Peter Harrington",
            "category": "科技",
            "isbn": "9787115317957",
            "publisher": "人民邮电出版社",
            "publish_date": "2013-06-01",
            "pages": 332,
            "price": 69.00,
            "description": "机器学习入门经典教材",
            "available": True,
            "total_copies": 4,
            "available_copies": 1,
            "location": "A区3层科技类书架"
        }
    }
    
    if book_id in books_detail:
        return json.dumps(books_detail[book_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "图书不存在"}, ensure_ascii=False)


def check_availability(book_id: str):
    """
    查询图书借阅状态
    
    Args:
        book_id: 图书ID
    """
    book_detail = json.loads(get_book_detail(book_id))
    if "error" in book_detail:
        return json.dumps({
            "error": "图书不存在",
            "book_id": book_id
        }, ensure_ascii=False)
    
    result = {
        "book_id": book_id,
        "title": book_detail["title"],
        "available": book_detail["available"],
        "total_copies": book_detail["total_copies"],
        "available_copies": book_detail["available_copies"],
        "location": book_detail["location"],
        "status": "可借" if book_detail["available_copies"] > 0 else "已借完"
    }
    
    return json.dumps(result, ensure_ascii=False)


def reserve_book(book_id: str, user_id: str):
    """
    预约借阅图书
    
    Args:
        book_id: 图书ID
        user_id: 用户ID
    """
    # 验证图书是否存在
    book_detail = json.loads(get_book_detail(book_id))
    if "error" in book_detail:
        return json.dumps({
            "error": "图书不存在",
            "book_id": book_id
        }, ensure_ascii=False)
    
    # 检查是否可借
    if book_detail["available_copies"] <= 0:
        return json.dumps({
            "error": "图书已全部借出，无法预约",
            "book_id": book_id,
            "title": book_detail["title"]
        }, ensure_ascii=False)
    
    # 生成预约ID
    reservation_id = f"res_{book_id}_{user_id}"
    
    result = {
        "reservation_id": reservation_id,
        "book_id": book_id,
        "book_title": book_detail["title"],
        "user_id": user_id,
        "status": "预约成功",
        "note": f"请在3天内到{book_detail['location']}办理借阅手续"
    }
    
    return json.dumps(result, ensure_ascii=False)


# ==================== 工具函数的JSON Schema定义 ====================

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": "搜索图书，支持按书名、作者、分类筛选",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "关键词（书名）"
                    },
                    "author": {
                        "type": "string",
                        "description": "作者姓名"
                    },
                    "category": {
                        "type": "string",
                        "description": "图书分类，如：小说、科技、历史、文学"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_book_detail",
            "description": "获取图书的详细信息，包括作者、出版社、价格、位置等",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "string",
                        "description": "图书ID，例如：book_001, book_002"
                    }
                },
                "required": ["book_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "查询图书是否可借，返回可借数量和位置信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "string",
                        "description": "图书ID"
                    }
                },
                "required": ["book_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reserve_book",
            "description": "预约借阅图书",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "string",
                        "description": "图书ID"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "用户ID"
                    }
                },
                "required": ["book_id", "user_id"]
            }
        }
    }
]


# ==================== Agent核心逻辑 ====================

available_functions = {
    "search_books": search_books,
    "get_book_detail": get_book_detail,
    "check_availability": check_availability,
    "reserve_book": reserve_book
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
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    messages = [
        {
            "role": "system",
            "content": """你是一位友好的图书管理员助手。你可以：
1. 帮助用户搜索图书（按书名、作者、分类）
2. 提供图书的详细信息
3. 查询图书的借阅状态
4. 帮助用户预约借阅图书

请根据用户的问题，使用合适的工具来获取信息并给出友好的回答。"""
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
        
        # 将模型响应加入对话历史
        messages.append(response_message)
        
        # 检查是否需要调用工具
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
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")
            
            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)
                
                print(f"工具返回: {function_response[:200]}..." if len(function_response) > 200 else f"工具返回: {function_response}")
                
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





if __name__ == "__main__":
    api_key = "sk-e829ed7689"
    
    # 示例1：搜索图书
    run_agent("我想找Python相关的书", api_key=api_key)
    
    # 示例2：查看图书详情
    run_agent("book_001这本书的详细信息是什么？", api_key=api_key)
    
    # 示例3：查询借阅状态
    run_agent("book_001这本书还能借吗？", api_key=api_key)
    
    # 示例4：预约借阅
    run_agent("我想借Python相关的书，用户ID是user_123", api_key=api_key)
    
    # 自定义查询
    # run_agent("你的问题", api_key=api_key)
