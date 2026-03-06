# Modification Info:
# Date: 2026-02-12
# Modified by: Huang Honghe
from typing import List, Union, Dict, Generator, Iterator
from datetime import datetime
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonAstREPLTool
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import re
import json
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine, text
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 只在release环境中导入telemetry
if os.environ.get("KOSTAL_ENV") == "release":
    from kostal_telemetry_client.client import KostalTelemetryClient
    os.environ["REQUESTS_CA_BUNDLE"] = "/app/kostal-ca/curl-ca-bundle.crt"
# ============ 配置常量类 ============

class Config:
    """集中管理所有配置常量和魔法数字"""

    # ===== 数据库配置 =====
    DB_HOST = os.environ.get('DB_HOST', 'cnsfjenl004.cn.kostal.int')
    DB_NAME = os.environ.get('DB_NAME', 'ai_related')
    DB_USER = os.environ.get('DB_USER', 'system')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', 'Kostal8888')

    # 数据库连接池配置
    DB_POOL_SIZE = 10
    DB_MAX_OVERFLOW = 20
    DB_POOL_RECYCLE_SECONDS = 3600  # 1小时后回收连接

    # ===== LLM 配置 =====
    LLM_API_KEY = os.environ.get('LLM_API_KEY', 'sk-ojnzuCVbMirKGKva6HpSdSbh2zSENhtKgQYiqReyqm1tBW8v')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://10.190.3.47:9997/v1')
    # 一阶主模型（用于常规意图识别）
    LLM_MODEL = os.environ.get('LLM_MODEL', 'qwen3')
    # 二阶小模型（用于二次咨询），默认与主模型相同，可通过环境变量单独配置
    LLM_SECOND_MODEL = os.environ.get('LLM_SECOND_MODEL', LLM_MODEL)
    LLM_TEMPERATURE = 0.0

    # ===== 对话管理配置 =====
    MAX_HISTORY_LENGTH = 10  # 对话历史最大长度
    RECENT_CONTEXT_LIMIT = 3  # 上下文中显示最近对话数量

    # ===== 重试配置 =====
    MAX_RETRIES = 10  # 查询最大重试次数

    # ===== 数据分页配置 =====
    CHUNK_SIZE = 25  # DataFrame分页显示大小

    # ===== 数据预览配置 =====
    PREVIEW_COLUMNS_COUNT = 3  # 预览显示的列数
    PREVIEW_LENGTH = 100  # 字符串预览长度
    QUERY_PREVIEW_LENGTH = 50  # 查询语句预览长度

    # ===== 风险分类阈值 =====
    RISK_HIGH_THRESHOLD = 0.8  # 高风险阈值
    RISK_MEDIUM_LOW_THRESHOLD = 0.5  # 中低风险分界线

    # ===== 数据转换默认值 =====
    DEFAULT_INT_FILL_VALUE = 0
    DEFAULT_FLOAT_FILL_VALUE = 0.0
    PERCENTAGE_DIVISOR = 100.0  # 百分比转小数除数
    UNKNOWN_INT_VALUE = -999  # 用于表示'Unknown'的整数标记值

    # ===== 长文本字段配置 =====
    LONG_TEXT_COLUMNS = ['Tessy_Summary', 'QAC_Summary', 'Jira_Overview', 'FS_Overview',
                         'SW_Release_Status', 'Base_On_Req1', 'Base_On_Req2', 'Base_On_TestCase',
                         'CS_Status', 'Project_Summary']
    LONG_TEXT_MAX_LENGTH = 2000  # 长文本字段的最大显示长度

    # ===== 数据库表配置 =====
    USER_QUERY_LOG_TABLE = 'user_query_log'

    # ===== DataFrame 列定义 =====
    # 整数列
    INT_COLUMNS = ['SYS_Testable', 'SYS_Passed', 'Issue_OpenNum', 'TotalRelease',
                   'SuccessRelease', 'Issue_TotalNum', 'Issue_OpenNum',
                   'Customer_Issue_TotalNum', 'Customer_Issue_OpenNum']

    # 百分比列（需要转换为浮点数）
    PERCENT_COLUMNS = ['VerifyPassRate', 'HIL&HILP_PassRate', 'RelSuccessRate', 'Issue_CloseRate',
                       'Customer_Issue_CloseRate']

    # ===== 环境配置 =====
    TELEMETRY_CA_BUNDLE_PATH = "/app/kostal-ca/curl-ca-bundle.crt"
    RELEASE_ENV_VALUE = "release"


class ConversationManager:
    """对话历史管理器 - 简化版，利用 Open-WebUI 的 messages 参数"""

    def __init__(self,max_history_length: int = 10):
        # 只保留 DataFrame 追问所需的数据
        self.last_query_result_full = None  # 上次查询结果完整数据（追问用）
        # 最大保留多少条对话记录
        self.max_history_length = max_history_length
        self.history = []

    def add_message(self, role: str, content: str):
        """向历史中添加一条消息，并按 max_history_length 截断"""
        self.history.append({"role": role, "content": content})

        # 如果超过最大长度，就只保留最后 N 条
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

    def clear_history(self):
        """清空历史"""
        self.history = []  

    def save_query_result(self, result, full_result=None, is_followup=False):
        """
        保存查询结果用于追问

        Args:
            result: 查询结果
            full_result: 完整结果 DataFrame
            is_followup: 是否为追问查询
        """
        # 保存最后的查询结果用于后续追问
        if isinstance(result, pd.DataFrame):
            # 关键修改：只有非追问查询才更新完整数据
            if not is_followup:
                # 保存完整版本用于追问
                if full_result is not None and isinstance(full_result, pd.DataFrame):
                    self.last_query_result_full = full_result.copy()
                else:
                    self.last_query_result_full = result.copy()  # 如果没有提供完整版，使用相同数据
            # 如果是追问查询，保持原有的 last_query_result_full 不变
        else:
            # 非DataFrame结果时，只有非追问查询才清空完整数据
            if not is_followup:
                self.last_query_result_full = None

    def has_previous_result_full(self):
        """检查是否有可用的完整上次查询结果"""
        return self.last_query_result_full is not None and not self.last_query_result_full.empty

    @staticmethod
    def extract_context_from_messages(messages: list, limit=None) -> str:
        # 从 Open-WebUI 传进来的整段 messages（user/assistant 消息列表）里，抽取最近几轮对话，生成一段“对话摘要字符串”，给分类器当上下文用。
        """
        从 Open-WebUI 的 messages 参数中提取上下文

        Args:
            messages: Open-WebUI 传入的消息列表
            limit: 限制提取的消息数量（默认最近3条）

        Returns:
            str: 格式化的上下文字符串
        """
        limit = limit or Config.RECENT_CONTEXT_LIMIT

        if not messages:
            return "无历史记录"

        # 只取最近的 limit*2 条消息（user+assistant 成对）
        recent_messages = messages[-(limit * 2):] if len(messages) > limit * 2 else messages

        context_lines = []
        i = 0

        # 遍历消息，将 user-assistant 配对
        while i < len(recent_messages):
            msg = recent_messages[i]

            if msg.get('role') == 'user':
                query = msg.get('content', '')
                query_preview = query[:50] + '...' if len(query) > 50 else query

                # 查找对应的 assistant 回复
                response = ''
                if i + 1 < len(recent_messages) and recent_messages[i + 1].get('role') == 'assistant':
                    response = recent_messages[i + 1].get('content', '')
                    i += 2  # 跳过 assistant 消息
                else:
                    i += 1

                # 生成响应摘要
                if '|' in response and 'ProjectNo' in response:
                    # DataFrame 格式的响应
                    lines = response.split('\n')
                    row_count = len([l for l in lines if '|' in l]) - 2  # 减去表头和分隔线
                    response_summary = f"包含{row_count}行数据，列：ProjectNo, Customer..."
                else:
                    # 文本响应
                    response_summary = response[:100] + '...' if len(response) > 100 else response

                # 标记最新的查询
                pair_index = len(context_lines) + 1
                total_pairs = (len(recent_messages) + 1) // 2
                marker = "【最新】" if pair_index == total_pairs else f"第{pair_index}次"

                context_lines.append(f"{marker} Q:{query_preview} -> {response_summary}")
            else:
                i += 1

        return "\n".join(context_lines) if context_lines else "无历史记录"

    @staticmethod
    def extract_project_from_messages(messages: list) -> str:
        """
        从 messages 中提取项目编号

        Args:
            messages: Open-WebUI 传入的消息列表

        Returns:
            str: 项目编号，如 'P08181'，未找到返回 None
        """
        import re

        # 从最近的消息向前查找
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # 尝试提取项目编号
                project_pattern = r'P\d{5}(?:\.\d+)?'
                match = re.search(project_pattern, content)
                if match:
                    return match.group(0)

        return None

    @staticmethod
    def extract_last_text_field_from_messages(messages: list) -> str:
        """
        从 messages 中提取上一次查询的文本字段类型

        Args:
            messages: Open-WebUI 传入的消息列表

        Returns:
            str: 字段名，如 'CS_Status'，未找到返回 None
        """
        # 定义各字段的特征关键词
        field_signatures = {
            'CS_Status': ['信息安全', '网络安全', 'CS状态','CS Overview总结','信息安全需求','测试失败的需求','Review占比'],
            'FS_Overview': ['功能安全', 'FS概览', 'FS状态',' 功能安全需求总数量',' 功能安全需求','FS Overview总结 ','FS总结'],
            'Jira_Overview': ['JIRA', 'Jira', 'jira', '总体JIRA概况', '客户端JIRA概况', 'JIRA Problem Found By分布', 'JIRA Change变更类型分布', '风险项'],
            'Base_On_Req1': ['系统需求覆盖率', '系统需求','客户需求',' 系统架构设计（System Architecture Design）',' 系统架构设计','软件需求','硬件需求','机械需求','风险项识别'],
            'Base_On_Req2': ['需求管理', '需求分析','Test Overview详情',' Test Overview总结','测试未覆盖的需求','测试结果不全的需求','测试失败的需求'],
            'Base_On_TestCase': ['测试用例', '测试用例分析',' BaseOnTestcase总结','HIL测试','HILP测试','HWT测试',' ITSW测试','UTSW测试','MECH测试','风险项'],
            'Project_Summary': ['项目总结', '项目概览', '项目概述', '整体状况','项目概况',
                '详细信息', '详情', '总结', '项目详情', '项目信息', '项目情况'],
            'Tessy_Summary': ['Tessy总结', 'Tessy概览','测试概况','测试总结','测试概览','测试概况总结','测试概述'],
            'SW_Release_Status': ['软件发布状态', '软件释放情况','SW_Release_Status总结',' Sample A','Sample B','Sample C','风险项'],
            'QAC_Summary': ['QAC测试总结', 'QAC测试概况','QAC静态代码检查报告总结','QAC','静态代码检查','静态代码'],
        }

        # 从最近的消息向前查找
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # 检测是哪个字段
                for field_name, keywords in field_signatures.items():
                    if any(kw in content for kw in keywords):
                        return field_name

        return None
class UserQueryLogger:
    """
    用户提问记录器 - 记录每次用户的问题到数据库
    """

    def __init__(self, db_connection_string: str = None):
        """
        Args:
            db_connection_string: SQLAlchemy 连接字符串，默认使用Config中的配置
        """
        if db_connection_string is None:
            db_connection_string = f"mysql+pymysql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}/{Config.DB_NAME}"
        self.db_connection_string = db_connection_string
        try:
            self.engine = create_engine(db_connection_string)
            with self.engine.connect() as conn:
                # 自动建表（如果不存在）
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {Config.USER_QUERY_LOG_TABLE} (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(100),
                        query TEXT,
                        classify VARCHAR(100),
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
            print("UserQueryLogger 初始化完成，已连接数据库。")
        except Exception as e:
            print(f"无法连接数据库或创建表: {e}")
            self.engine = None

    def log(self, username: str, query: str, classify: str = None):
        """
        记录用户查询
        Args:
            username: 用户名
            query: 用户输入的问题
            classify: 模型分类结果（如 NormalTemplate、FollowUpTemplate）
        """
        if self.engine is None:
            print("[警告] 数据库未初始化，跳过日志记录。")
            return

        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(f"INSERT INTO {Config.USER_QUERY_LOG_TABLE} (username, query, classify) VALUES (:u, :q, :c)"),
                    {"u": username, "q": query, "c": classify or "unknown"}
                )
                conn.commit()
        except Exception as e:
            print(f"[警告] 写入日志失败: {e}")

class DB_Reader:
    def __init__(self):
        self.connection_string = f"mysql+pymysql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}/{Config.DB_NAME}"

        # 创建带连接池的 engine，复用连接
        self.engine = create_engine(
            self.connection_string,
            pool_size=Config.DB_POOL_SIZE,
            max_overflow=Config.DB_MAX_OVERFLOW,
            pool_pre_ping=True,     # 每次从池中获取连接时先ping，确保连接有效
            pool_recycle=Config.DB_POOL_RECYCLE_SECONDS,
            echo=False              # 不打印SQL日志
        )
        print(f"DB_Reader 初始化完成，连接池已创建 (pool_size={Config.DB_POOL_SIZE}, max_overflow={Config.DB_MAX_OVERFLOW})")
        
        # 指标收集
        self.query_times = []
        self.total_queries = 0

    def query(self, query: str) -> pd.DataFrame:
        """执行SQL查询并返回DataFrame，复用连接池中的连接"""
        start_time = time.time()
        try:
            # 复用self.engine，不再每次创建新engine
            df = pd.read_sql(query, self.engine)

            # 数据类型转换
            self.safe_str_to_int(df, Config.INT_COLUMNS,
                                fill_value=Config.DEFAULT_INT_FILL_VALUE, inplace=True)
            self.safe_percent_to_float(df, Config.PERCENT_COLUMNS,
                                      fill_value=Config.DEFAULT_FLOAT_FILL_VALUE, inplace=True)
            
            # 记录查询时间
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            self.total_queries += 1
            
            return df
        except Exception as e:
            print(f"[警告] 数据库查询失败: {e}")
            raise

    def close(self):
        """关闭数据库连接池，释放资源"""
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()
            print("DB_Reader 连接池已关闭")

    def get_metrics(self):
        """获取数据库查询指标"""
        if self.total_queries == 0:
            return {
                "total_queries": 0,
                "avg_query_time": 0.0,
                "total_query_time": 0.0
            }
        return {
            "total_queries": self.total_queries,
            "avg_query_time": sum(self.query_times) / len(self.query_times),
            "total_query_time": sum(self.query_times)
        }

    def __del__(self):
        """析构函数，确保连接池被正确关闭"""
        self.close()

    def safe_str_to_int(self, df: pd.DataFrame,
                   columns: Union[str, List[str]],
                   fill_value: int = None,
                   remove_whitespace: bool = True,
                   inplace: bool = False) -> pd.DataFrame:
        """安全地将DataFrame中的字符串列转换为整数类型"""
        if fill_value is None:
            fill_value = Config.DEFAULT_INT_FILL_VALUE

        if not inplace:
            df = df.copy()

        if isinstance(columns, str):
            columns = [columns]

        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"列不存在: {missing_cols}")

        for col in columns:
            original_dtype = df[col].dtype
            original_non_null_count = df[col].notna().sum()

            if remove_whitespace and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()

            df[col] = df[col].replace(['', 'None', 'none', 'NULL', 'null'], np.nan)
            df[col] = df[col].replace(['Unknown', 'unknown'], Config.UNKNOWN_INT_VALUE)
            df[col] = pd.to_numeric(df[col], errors='coerce')

            nan_count = df[col].isna().sum()
            invalid_count = nan_count - (len(df) - original_non_null_count)

            if nan_count > 0:
                df[col] = df[col].fillna(fill_value)
            
            df[col] = df[col].astype(int)
        
        return df if not inplace else None
    
    def safe_percent_to_float(self, df: pd.DataFrame,
                         columns: Union[str, List[str]],
                         fill_value: float = None,
                         remove_whitespace: bool = True,
                         inplace: bool = False) -> pd.DataFrame:
        """安全地将DataFrame中的百分比字符串列转换为浮点数类型"""
        if fill_value is None:
            fill_value = Config.DEFAULT_FLOAT_FILL_VALUE

        if not inplace:
            df = df.copy()
        
        if isinstance(columns, str):
            columns = [columns]
        
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"列不存在: {missing_cols}")
        
        for col in columns:
            original_dtype = df[col].dtype
            original_non_null_count = df[col].notna().sum()
            
            if remove_whitespace and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
            
            invalid_values = ['', 'None', 'none', 'NULL', 'null', 'Unknown', 'unknown', 'N/A', 'n/a', 'NA', 'na']
            df[col] = df[col].replace(invalid_values, np.nan)
            
            def convert_percent_to_float(value):
                if pd.isna(value):
                    return np.nan
                
                try:
                    str_value = str(value).strip()
                    match = re.match(r'^(-?\d*\.?\d+)%?$', str_value)
                    
                    if match:
                        number = float(match.group(1))
                        if '%' in str_value:
                            return number / Config.PERCENTAGE_DIVISOR
                        else:
                            return number
                    else:
                        return np.nan
                        
                except (ValueError, AttributeError):
                    return np.nan
            
            df[col] = df[col].apply(convert_percent_to_float)
            
            nan_count = df[col].isna().sum()
            invalid_count = nan_count - (len(df) - original_non_null_count)
            
            if nan_count > 0:
                df[col] = df[col].fillna(fill_value)
            
            df[col] = df[col].astype(float)
        
        return df if not inplace else None

    def safe_str_to_float(self, df: pd.DataFrame, 
                        columns: Union[str, List[str]], 
                        fill_value: float = 0.0,
                        decimal_places: int = None,
                        inplace: bool = False) -> pd.DataFrame:
        """通用的字符串转浮点数函数（不专门处理百分比）"""
        
        if not inplace:
            df = df.copy()
        
        if isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            # 清理数据
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['', 'None', 'none', 'NULL', 'null', 'Unknown', 'unknown'], np.nan)
            
            # 转换为数字
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 填充NaN
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                df[col] = df[col].fillna(fill_value)
            
            # 保留小数位数
            if decimal_places is not None:
                df[col] = df[col].round(decimal_places)
            
            # 确保是float类型
            df[col] = df[col].astype(float)
        
        return df

# 全局数据库读取器实例
db_reader = DB_Reader()

# 懒加载：不在模块导入时立即查询，而是在首次访问时才加载
_products_df = None

def get_products_df():
    """懒加载 products_df，首次调用时才从数据库查询"""
    global _products_df
    if _products_df is None:
        print("[正在加载] 首次加载项目数据...")
        _products_df = db_reader.query("SELECT * FROM Project_List")
        print(f"[成功] 项目数据加载完成，共 {len(_products_df)} 行")
    return _products_df

# ============ 提示模板基础组件 ============
# 列描述（所有模板共用）
column_descriptions = """
结构化数据列（可直接用于筛选和统计）：
- id: 主键（整数）
- Department: 部门（字符串，如 AE1, AE2, AE3, AE4）
- ProjectNo: 项目编号（字符串，如 P08181）
- Customer: 客户名称（字符串，如 JAC, VOLVO）
- Carline: 车型（字符串）
- ProductVariant: 产品变种（字符串）
- SamplePhase: 样件阶段（字符串，如 A01, B01, C01）
- SYS_Testable: 可测试的系统需求数量（整数）
- SYS_Passed: 系统需求测试通过数量（整数）
- VerifyPassRate: 系统需求验证通过率（浮点数，0-1）
- HIL&HILP_PassRate: HIL&HILP测试通过率（浮点数，0-1）
- TotalRelease: 样件发布总次数（整数）
- SuccessRelease: 样件发布成功次数（整数）
- RelSuccessRate: 样件发布成功率（浮点数，0-1）
- Issue_TotalNum: JIRA问题总数（整数）
- Issue_OpenNum: 未关闭的JIRA问题数（整数）
- Issue_CloseRate: JIRA问题关闭率（浮点数，0-1）
- Customer_Issue_TotalNum: 客户相关JIRA问题总数（整数）
- Customer_Issue_OpenNum: 未关闭的客户相关JIRA问题数（整数）
- Customer_Issue_CloseRate: 客户问题关闭率（浮点数，0-1）
- Date: 数据日期（日期，如：2025-12-23 13:41:52）
- CalendarWeek: 日历周（字符串）
- Risk_Score_Latest: 最新风险评分（浮点数，0-1）
- Risk_Score_LSTM: 基于LSTM模型的风险评分（浮点数，0-1）
- Data_Consistent_Check: 数据一致性检查结果（字符串）

非结构化数据列（长文本，需要 LLM 理解）：
- Project_Summary: 项目总结（长文本）
- CS_Status: 信息安全状态（长文本）
- Jira_Overview: JIRA问题概览（长文本）
- FS_Overview: 功能安全概览（长文本）
- Tessy_Summary: Tessy测试总结（长文本）
- QAC_Summary: QAC静态代码检查总结（长文本）
- SW_Release_Status: 软件发布状态（长文本）
- Base_On_Req1: 需求信息总结1（长文本）
- Base_On_Req2: 需求信息总结2（长文本）
- Base_On_TestCase: 测试用例总结（长文本）
"""


# 共同的关键规则（所有模板共用）
base_rules = """
关键规则 - 必须严格遵守，否则会报错：

1. 多条件查询必须使用 & 和 | 符号，绝对不能使用 and 和 or
2. 每个条件必须用圆括号包裹（因为运算符优先级）
3. 正确格式：df[(df['列名1'] == '值1') & (df['列名2'] > 值2)]

**错误写法（会导致系统错误）：**
- df[(df['col1'] == 'val1') and (df['col2'] > 5)]     ← 使用了 and
- df[df['col1'] == 'val1' & df['col2'] > 5]      ← 缺少括号，优先级错误

**正确写法：**
- df[(df['col1'] == 'val1') & (df['col2'] > 5)]
"""

# 共同的基础指令（所有模板共用）
base_instructions = """
你是一个项目管理专家，根据用户问题生成Pandas查询代码。遵循以下规则：
1. 始终使用标准的DataFrame查询语法。**不要思考过程**。
2. 列名使用单引号包裹（如 df['age']）
3. 结果必须是DataFrame操作表达式（如 df.query(...) 或 df[...]）输出只包含df表达式，不包含其他任何东西
4. df查询的列的名字必须是 {Columnname}里面的，不允许臆测，问题{question}中没有的的列不要加入表达式中。
5. df['列名']中，禁止使用未定义的列，必须使用{Columnname}中的列
6. **重要：大小写不敏感匹配规则（仅针对Customer列）**
   - 对于Customer列的匹配，必须使用大小写不敏感的匹配
   - 使用 .str.strip().str.lower() 进行转换后再比较，确保匹配所有大小写形式和去除前后空格
   - 错误示例：df[df['Customer'] == 'Volvo'] ❌ (只能匹配"Volvo"，无法匹配"VOLVO"、"volvo"、"Volvo "等)
   - 错误示例：df[df['Customer'].str.lower() == 'volvo'] ❌ (无法匹配"Volvo "、" VOLVO"等带空格的)
   - 正确示例：df[df['Customer'].str.strip().str.lower() == 'volvo'] ✅ (可以匹配"Volvo"、"VOLVO"、"volvo"、"Volvo "、" VOLVO"等所有形式)
   - 注意：此规则仅适用于Customer列，其他列（如Department、ProjectNo等）使用标准的大小写敏感匹配
7. 样件阶段分为Axx、Bxx、Cxx阶段，当问及样件阶段是需要对xx做通配处理：
   - A样件阶段：使用 df['SamplePhase'].str.startswith('A')
   - B样件阶段：使用 df['SamplePhase'].str.startswith('B')
   - C样件阶段：使用 df['SamplePhase'].str.startswith('C')
   - 具体阶段如A01：使用 df['SamplePhase'] == 'A01'
7. **重要：Unknown值匹配规则（仅针对整数列）**
   - 对于整数列（如TotalRelease、SuccessRelease、SYS_Testable、SYS_Passed、RelSuccessRate等），当用户查询"Unknown"、"是Unknown"、"为Unknown"等时，使用 -999 来匹配
   - 示例：
     * "TotalRelease是Unknown的项目" → df[df['TotalRelease'] == -999][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'TotalRelease']]
     * "SuccessRelease为Unknown的项目" → df[df['SuccessRelease'] == -999][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'SuccessRelease']]
   - 注意：此规则仅适用于整数列，字符串列（如Customer、Department等）不适用此规则
8. 百分比转换规则：
   - 识别格式：30%、百分之三十、30个百分点
   - 转换方式：除以100得到小数（如 30% → 0.3）
   - 比较操作：
     * "大于30%" → > 0.3
     * "超过30%" → > 0.3
     * "30%以上" → >= 0.3
     * "低于30%" → < 0.3
     * "30%以下" → <= 0.3
     * "30%到50%" → >= 0.3 & <= 0.5
9. 分位数查询规则（前N%、后N%）：
   - "前10%"：使用 quantile(0.9) 计算90%分位数，筛选大于等于该值的记录
   - "后10%"：使用 quantile(0.1) 计算10%分位数，筛选小于等于该值的记录
   - "前20%"：使用 quantile(0.8)
   - "后30%"：使用 quantile(0.3)
   - 示例：
     * "测试通过率前10%的项目" → df[df['VerifyPassRate'] >= df['VerifyPassRate'].quantile(0.9)][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'VerifyPassRate']]
     * "测试通过率后10%的项目" → df[df['VerifyPassRate'] <= df['VerifyPassRate'].quantile(0.1)][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'VerifyPassRate']]
     * "风险评分前20%的项目" → df[df['Risk_Score_Latest'] >= df['Risk_Score_Latest'].quantile(0.8)][['Department','ProjectNo', 'Customer', 'Risk_Score_Latest']]
10. 查询特定列时，使用双层方括号返回DataFrame而不是Series，如 df[条件][['列名']] 而不是 df[条件]['列名']
10. 查询结果中必须包含查询条件列，如df[df['Customer'].str.lower() == 'volvo'][['Department','ProjectNo', 'Customer', 'ProductVariant','Carline']]，结果中必须包含Department列
11. 当无法理解问题时，主动引导用户提供更多信息

【重要】输出格式要求：
- 只输出纯Python代码，不要包含任何解释、注释或标签
- 不要包含</think>标签<think>
- 不要包含```python或```代码块标记
- 直接输出可执行的DataFrame表达式

**重要：groupby操作规则**
11. 使用groupby时，必须在最后添加.reset_index()，确保分组列显示在结果中
12. 统计查询时，给聚合列重命名，使结果更清晰

**重要：极值查询规则**
13. 当用户查询"最大"、"最小"、"最多"、"最少"、"最高"、"最低"等极值时：
    - 不要使用 .head(1) 或 .sort_values().head(1)
    - 必须返回所有极值相同的记录
    - 使用 df[df['列名'] == df['列名'].max()] 或 df[df['列名'] == df['列名'].min()]
    - 示例：
      * "哪个项目风险最高？" → max_risk = df['Risk_Score_Latest'].max(); df[df['Risk_Score_Latest'] == max_risk][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'Risk_Score_Latest']]
      * "哪个项目的当前风险最高？" → max_risk = df['Risk_Score_Latest'].max(); df[df['Risk_Score_Latest'] == max_risk][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'Risk_Score_Latest', 'Project_Summary']]
      * "客户问题单最多的项目" → df[df['Customer_Issue_TotalNum'] == df['Customer_Issue_TotalNum'].max()][['ProjectNo', 'Customer', 'Carline', 'Department', 'ProductVariant', 'Customer_Issue_TotalNum']]
      * "AE1部门测试通过率最高的项目" → max_rate = df[df['Department'] == 'AE1']['VerifyPassRate'].max(); df[(df['Department'] == 'AE1') & (df['VerifyPassRate'] == max_rate)][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'VerifyPassRate']]

**重要：统计数量规则**
13. 当问"有多少个项目"、"一共有多少个"、"有几个"等统计数量问题时：
   - 返回符合条件的DataFrame，不要使用shape[0]、len()或.count()返回整数
   - 用户可以通过查看返回的DataFrame行数来了解数量
   - 示例：
     * "AE2部门有多少个项目？" → df[df['Department'] == 'AE2'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline']]
     * "Volvo有几个项目？" → df[df['Customer'].str.lower() == 'volvo'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline']]
   - 注意：一个项目编号可能对应多行记录（不同样件阶段、产品变体等），返回所有记录

**重要：列选择规则**
- 默认核心列：Department, ProjectNo, Customer, ProductVariant, Carline, Project_Summary
- 查询结果必须包含 Department 和 ProjectNo 列
- 根据用户问题添加相关列：
  * 问风险 → 添加 Risk_Score_Latest
  * 问测试 → 添加 VerifyPassRate, SYS_Testable, SYS_Passed
  * 问发布 → 添加 RelSuccessRate, TotalRelease, SuccessRelease
  * 问问题 → 添加 Issue_TotalNum, Issue_OpenNum, Issue_CloseRate
  * 问项目基本信息（如"P08486的基本信息"、"查询项目P08760"、"P08878的项目详情"、"显示P09003的信息"、"P08910项目的情况"等）→ 添加 Project_Summary
- 只有用户明确要求"所有信息"或"全部列"时，才返回所有列
- 查询条件列必须包含在结果中（如按Department筛选，结果必须包含Department列）

**重要：行数限制规则**
- 不要使用 .head()、.tail() 或 .sample() 来限制返回的行数，除非用户明确要求"前N个"、"最后N个"或"随机N个"
- 查询"最高/最低/最大/最小"等极值时，使用极值查询规则（规则13），不要使用排序
- 查询"有几个"、"有多少"、"多少个"时，返回符合条件的DataFrame，不要使用len()或.count()返回整数
- 示例：
  * "风险评分最高的前3个项目" → df.sort_values('Risk_Score_Latest', ascending=False).head(3)[['Department','ProjectNo', 'Customer', 'Risk_Score_Latest']]
  * "AE1部门有几个项目？" → df[df['Department'] == 'AE1'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline']]
"""

# ============ 具体的提示模板 ============

normal_prompt = PromptTemplate.from_template(base_rules + base_instructions + """
**重要**：禁止使用 .T 转置，只返回指定的列，不要返回所有列，除非用户明确要求"所有信息"或"全部列"
                                             
    {column_descriptions}

    示例：
        输入: AE1部门且风险评分大于0.5的项目
	    输出: df[(df['Department'] == 'AE1') & (df['Risk_Score_Latest'] > 0.5)][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'Risk_Score_Latest']]

        输入: AE1部门有哪些项目？
        输出: df[df['Department'] == 'AE1'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline']]

        输入: 查询项目P08379的基本信息
        输出: df[df['ProjectNo'] == 'P08379'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'Project_Summary']].copy().assign(Project_Summary=df[df['ProjectNo'] == 'P08379']['Project_Summary'].str[:2000])
        
        输入: 查询项目P08379的详细信息
        输出: df[df['ProjectNo'] == 'P08379'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'Project_Summary']].copy().assign(Project_Summary=df[df['ProjectNo'] == 'P08379']['Project_Summary'].str[:2000])
        
        输入: 显示项目P08379的详情
        输出: df[df['ProjectNo'] == 'P08379'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'Project_Summary']].copy().assign(Project_Summary=df[df['ProjectNo'] == 'P08379']['Project_Summary'].str[:2000])

        输入: 显示P09003的信息
        输出: df[df['ProjectNo'] == 'P09003'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'Project_Summary']].copy().assign(Project_Summary=df[df['ProjectNo'] == 'P09003']['Project_Summary'].str[:2000])

        输入: 查询所有项目的 Tessy_Summary？
        输出:  df[['Department','ProjectNo', 'Tessy_Summary']]
        
        输入: 查询所有项目的 SW_Release_Status？
        输出: 输出: df[['Department','ProjectNo', 'SW_Release_Status']]

        输入: 按风险评分降序排列所有项目
        输出: df.sort_values('Risk_Score_Latest', ascending=False)[['Department','ProjectNo', 'Customer', 'Risk_Score_Latest']]

        输入: AE1部门测试通过率最高的项目
        输出: max_rate = df[df['Department'] == 'AE1']['VerifyPassRate'].max(); df[(df['Department'] == 'AE1') & (df['VerifyPassRate'] == max_rate)][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'VerifyPassRate']]

        输入: 查询客户名称包含"JAC"的项目
        输出: df[df['Customer'].str.strip().str.lower().str.contains('jac', case=False)][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline']]

        输入: 查询风险评分大于0.8且客户是JAC的项目
        输出: df[(df['Risk_Score_Latest'] > 0.8) & (df['Customer'].str.strip().str.lower() == 'jac')][['Department','ProjectNo', 'Customer', 'Risk_Score_Latest','Project_Summary']]

        输入: 查询A样件阶段的所有项目
        输出: df[df['SamplePhase'].str.startswith('A')][['Department','ProjectNo', 'Customer', 'SamplePhase','Project_Summary']]

        输入: 统计每个客户的平均风险评分
        输出: df.groupby('Customer')['Risk_Score_Latest'].mean().reset_index().rename(columns={{'Risk_Score_Latest': '平均风险评分'}})

        输入: 查询验证通过率大于80%的项目
        输出: df[df['VerifyPassRate'] > 0.8][['Department','ProjectNo', 'Customer', 'VerifyPassRate','Project_Summary']]

        输入: 测试通过率前10%的项目
        输出: df[df['VerifyPassRate'] >= df['VerifyPassRate'].quantile(0.9)][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'VerifyPassRate']]

        输入: 测试通过率后10%的项目
        输出: df[df['VerifyPassRate'] <= df['VerifyPassRate'].quantile(0.1)][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'VerifyPassRate']]

        输入: 风险评分前20%的项目
        输出: df[df['Risk_Score_Latest'] >= df['Risk_Score_Latest'].quantile(0.8)][['Department','ProjectNo', 'Customer', 'Risk_Score_Latest']]

        输入: AE1部门有几个项目？
        输出: df[df['Department'] == 'AE1'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline']]

        输入: HKMC客户的项目有几个？
        输出: df[df['Customer'].str.strip().str.lower() == 'hkmc'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline']]

        输入: 这些项目中有多少个风险评分大于0.5？
        输出: df[df['Risk_Score_Latest'] > 0.5][['Department','ProjectNo', 'Customer', 'Risk_Score_Latest']]
        
    当前用户问题：{question}
    上次的运行结果:{lastcode},你需要纠正这个错误！
    """, partial_variables={
        "column_descriptions": column_descriptions
    })

detail_info_prompt = PromptTemplate.from_template(base_rules + """
**特殊规则**：询问详情/状态/概述/摘要时：
- 如果用户明确指定了字段（如 Tessy_Summary、CS_Status、Jira_Overview 等），返回该指定字段
- 如果用户只说"详细信息"、"详情"、"总结"、"项目详情"、"项目信息"、"项目情况"等通用词汇，默认返回 Project_Summary
- 所有长文本字段最多显示2000字符
**重要**：返回结果必须包含Department和ProjectNo列
**重要**：只返回核心列（Department, ProjectNo, Customer, ProductVariant, Carline）和相关的文本字段（如Project_Summary），不要返回所有列，除非用户明确要求"所有信息"或"全部列"
""" + base_instructions + """

    {column_descriptions}

    示例：
        输入: 项目P08379的详细信息
        输出: df[df['ProjectNo'] == 'P08379'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline']].copy().assign(Project_Summary=df[df['ProjectNo'] == 'P08379']['Project_Summary'].str[:2000])

        输入: AE1部门且风险大于0.5的项目详情
        输出: df[(df['Department'] == 'AE1') & (df['Risk_Score_Latest'] > 0.5)][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'Risk_Score_Latest']].copy().assign(Project_Summary=df[(df['Department'] == 'AE1') & (df['Risk_Score_Latest'] > 0.5)]['Project_Summary'].str[:2000])

        输入: 查询所有项目的 Tessy_Summary？
        输出: df[['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline']].copy().assign(Tessy_Summary=df['Tessy_Summary'].str[:2000])

    当前用户问题：{question}
    上次的运行结果:{lastcode},你需要纠正这个错误！
""", partial_variables={
    "column_descriptions": column_descriptions
})

risk_info_prompt = PromptTemplate.from_template(base_rules + f"""
**特殊规则**：风险分类
- 风险系数大于{Config.RISK_HIGH_THRESHOLD}为高风险
- 风险系数在{Config.RISK_MEDIUM_LOW_THRESHOLD}到{Config.RISK_HIGH_THRESHOLD}为中风险
- 风险系数小于{Config.RISK_MEDIUM_LOW_THRESHOLD}为低风险

**特殊规则**：各项指标都很差的定义
- VerifyPassRate 低于 60% (即 < 0.6)
- HIL&HILP_PassRate 低于 60% (即 < 0.6)
- Risk_Score_Latest 高于 0.5 (即 > 0.5)

""" + base_instructions + """

{column_descriptions}

示例：
输入: AE1部门的中风险项目
输出: df[(df['Department'] == 'AE1') & (df['Risk_Score_Latest'] >= 0.5) & (df['Risk_Score_Latest'] <= 0.8)][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'Risk_Score_Latest','Project_Summary']]

输入: 各项指标都很差的项目有哪些？
输出: df[(df['VerifyPassRate'] < 0.6) & (df['HIL&HILP_PassRate'] < 0.6) & (df['Risk_Score_Latest'] > 0.5)][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'VerifyPassRate', 'HIL&HILP_PassRate', 'Risk_Score_Latest', 'Project_Summary']]

输入: 哪个项目的当前风险最高？
输出: max_risk = df['Risk_Score_Latest'].max(); df[df['Risk_Score_Latest'] == max_risk][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline', 'Risk_Score_Latest', 'Project_Summary']]

当前用户问题：{question}
上次的运行结果:{lastcode},你需要纠正这个错误！
""", partial_variables={
    "column_descriptions": column_descriptions
})

# 改进的追问查询模板（基于上次查询结果）
# 注意：使用 last_df 而不是 df
followup_base_rules = base_rules.replace("df[", "last_df[").replace("使用 df", "使用 last_df")

followup_prompt = PromptTemplate.from_template(followup_base_rules + """
你是一个项目管理专家，用户正在基于上次查询结果进行追问。

{column_descriptions}

上下文信息：
{context_info}

遵循以下规则：
1. 始终使用标准的DataFrame查询语法。不要思考过程。
2. 列名使用单引号包裹（如 last_df['age']）
3. 基于上次结果DataFrame(变量名：last_df)进行操作
4. 结果必须是DataFrame操作表达式，输出只包含表达式，不包含其他任何东西
5. last_df['列名']中，禁止使用未定义的列
6. 查询特定列时，使用双层方括号返回DataFrame而不是Series
7. 如果问题涉及筛选、排序、统计等操作，直接在last_df上操作

# 特殊处理规则：
8. 当用户询问"这个项目"、"该项目"、"此项目"的信息时：
   - 如果last_df只有一行数据，直接返回该行的相关列
   - 如果last_df有多行数据，需要用户明确指定具体哪个项目

9. 当用户询问项目的详细信息、总结、摘要时，包含以下列：
   - 必选：ProjectNo, ProductVariant, Project_Summary
   - 可选：Customer, Department, Carline（根据查询上下文）

10. 当用户询问"平均"、"总计"、"最大"、"最小"等统计值时：
    - 平均值：使用 last_df['列名'].mean()
    - 总计：使用 last_df['列名'].sum()
    - 最大值：使用 last_df['列名'].max()
    - 最小值：使用 last_df['列名'].min()
    - 当用户询问"有几个"、"有多少"、"多少个"时，返回符合条件的DataFrame，不要使用len()或.count()返回整数

11. 追问示例：
    - "这些项目中风险最高的是哪个？" → last_df[last_df['Risk_Score_Latest'] == last_df['Risk_Score_Latest'].max()][['ProjectNo', 'Risk_Score_Latest']]
    - "其中HKMC客户的项目有几个？" → last_df[last_df['Customer'].str.strip().str.lower() == 'hkmc'][['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline']]
    - "这些项目按风险评分排序" → last_df.sort_values('Risk_Score_Latest', ascending=False)[['Department','ProjectNo', 'Customer', 'Risk_Score_Latest']]
    - "这些项目中测试通过率前10%的项目" → last_df[last_df['VerifyPassRate'] >= last_df['VerifyPassRate'].quantile(0.9)][['Department','ProjectNo', 'Customer', 'VerifyPassRate']]
    - "这些项目中测试通过率后20%的项目" → last_df[last_df['VerifyPassRate'] <= last_df['VerifyPassRate'].quantile(0.2)][['Department','ProjectNo', 'Customer', 'VerifyPassRate']]

【重要】输出格式要求：
- 只输出纯Python代码，不包含任何解释、注释或标签
- 不要包含<thinking>、</thinking>、<think>、</think>标签
- 不要包含```python或```代码块标记
- 直接输出可执行的DataFrame表达式

可用的数据列：{available_columns}

当前追问：{question}
上次的运行结果:{lastcode},你需要纠正这个错误！
""", partial_variables={
    "column_descriptions": column_descriptions
})

# 文本字段理解模板 - 用于从大文本字段中提取信息
text_understanding_template = PromptTemplate.from_template("""
你是一个专业的数据分析助手，需要从项目的详细文本信息中提取用户需要的数据。

**项目编号**：{project_no}

**文本字段内容**：
{text_content}

**用户问题**：
{question}

**任务要求**：
1. 上述文本内容是项目{project_no}的{field_name}字段数据
2. 仔细阅读上述文本内容{text_content}
3. 根据用户问题{question}，提取相关的具体信息
4. 仔细分析并用简洁、准确的自然语言回答
5. 针对需要计算的问题，需要直接返回原始文本数据

**回答格式指南**：
- 如果问题是询问**详细信息、项目详情、完整情况**等需要全面了解项目的问题（如"详细信息"、"项目详情"、"项目情况"等），需要全面、系统地总结文本内容的各个方面
  示例："项目P08181的详细情况如下：\n1. 需求实现：客户需求实现率98.33%，系统需求实现率98.50%\n2. 测试情况：可测需求测试通过率84.8%，HIL测试执行率100%\n3. JIRA问题：KOCHI AE Issue-Problem未关闭46个，全部逾期\n4. 信息安全：信息安全需求共216条，Review占比26.39%\n5. 样件放行：Sample C放行成功率54.55%\n6. 建议：加强JIRA问题处理效率，补充系统需求测试用例"

- 如果问题是询问**具体数值**（如"JIRA问题总数是多少"），需要仔细理解总体JIRA概况的文本，并给出判断结果和依据。
  示例："JIRA问题总数为150“

- 如果问题是**是否判断**（如"Review占比是否超过30%"），给出判断结果和依据
  示例："Review占比为29.4%，未超过30%的警戒线"

- 如果问题是**列表提取**（如"测试失败的需求有哪些"），列出具体项目
  示例："测试失败的需求共4条：\n1. SYS_Nissan_CS_649\n2. SYS_Nissan_CS_650\n3. SYS_Nissan_CS_653\n4. SYS_Nissan_CS_655"

- 如果问题是**分析建议**（如"主要问题是什么"），提炼关键信息并给出分析
  示例："主要问题包括：Review未通过11条，测试结果不全15条，测试失败4条。建议优先处理Review未通过的需求"

**重要**：
- 直接回答问题，不要重复问题本身
- 答案必须基于文本内容，不要编造信息
- 如果文本中没有相关信息，明确说明"文本中未包含相关信息"
- 对于"详细信息"、"项目详情"等需要全面了解的问题，可以适当增加内容的详细程度；其他问题保持简洁（一般3-5句话）
""")



# 分类器定义
System_prompt_text_class_x = """
你是一个意图识别专家,需要先分步思考（输出推理过程），再得到分类类别的分数，然后取出分数最高的类别Classify, 使用 classify 来区分问题是属于哪种类型。

## 思维链（CoT）推理要求
你需要先按照以下步骤完整思考并输出推理过程，再生成最终的JSON结果：
步骤1：提取问题核心特征
   - 列出问题中的关键信息、关键词、指代关系（如有）、上下文关联（如有）。
步骤2：逐一匹配分类规则
   - 针对每个可选类别（HelpTemplate/Other_template/NormalTemplate/DetailInfoTemplate/RiskRateTemplate/RiskDetermination/FollowUpTemplate/ContinueTemplate），分析问题是否符合该类别的规则，明确“符合/不符合”的具体依据。
步骤3：计算得分的逻辑说明
   - 说明每个类别得分的计算依据（如“因完全匹配NormalTemplate规则，得分0.9；Other_template无匹配点，得分0.0”），确保所有得分之和为1。
步骤4：最终分类确认
   - 说明选择最终classify的理由（如“NormalTemplate得分最高（0.9），因此classify为NormalTemplate”）。

## 分类规则（按优先级理解与应用，但输出时需给出每个类型的得分以及最终一种类型的 classify 值）：
# 1. Other_template
    - 若问题与"项目管理"无关，classify it as 'Other_template'。
    - 与项目管理无关，指无法合理映射到：项目、样件、客户、风险、需求等项目管理语境。
# 1.1 HelpTemplate（帮助请求）
    - 若问题是关于"帮助、使用说明、功能介绍"等，classify it as 'HelpTemplate'。
    - 示例：
    * "帮助"
    * "你能做什么？"
    * "怎么查询项目？"
    * "使用说明"
    * "功能介绍"
    * "如何使用"等
# 2. NormalTemplate
    - 若问题是关于"项目、样件、客户"相关的内容，且为新的独立查询，并且：
    * 不需要理解大文本字段（如概览、总结、详细说明等），
    * 仅涉及基本字段、列表、简单统计，
    则 classify it as 'NormalTemplate'。
    - 示例（NormalTemplate）：
    ✅ "AE1部门有哪些项目？" → NormalTemplate（列表查询）
    ✅ "2025年12月的数据" → NormalTemplate（日期筛选查询）
    ✅ "CalendarWeek是2025-W52的项目" → NormalTemplate（日历周筛选查询）
    ✅ "最近更新的项目" → NormalTemplate（日期排序查询）
    ✅ "客户是JAC的项目有哪些？" → NormalTemplate（客户筛选查询）
# 3. DetailInfoTemplate（文本字段详细信息分类——需要 LLM 理解文本内容的查询）
    - 若问题涉及以下长文本字段的“详细内容理解”，classify it as 'DetailInfoTemplate'：
    * Project_Summary：项目总结、项目概览、整体状况
    * CS_Status / CS_Status_Detail：信息安全、网络安全、CS状态
    * Jira_Overview / Jira_Detail：JIRA 问题、问题关闭率、问题详情
    * FS_Overview / FS_Detail：功能安全概览、功能安全状态、功能安全详情
    * Base_On_Req1 / Base_On_Req2：需求覆盖率、需求管理、需求分析
    * Tessy_Summary / Tessy_Detail：Tessy测试总结、Tessy测试详情
    * QAC_Summary / QAC_Detail：QAC测试总结、QAC测试详情
    * SW_Release_Status：软件发布状态
    * Base_On_TestCase：基于测试用例的分析
    - 关键识别词（只要与上述长文本字段语义有关，即可触发 DetailInfoTemplate）：
    * 详细信息相关：
        - 详情、总结、概览、状况、建议、主要问题、内容是什么
    * 文本内容分析相关：
        - 有哪些、有多少、是否包含、提到了什么、关于...的内容
    * 特定字段相关：
        - CS、信息安全、JIRA、FS、功能安全、需求覆盖、Tessy、QAC、发布状态等
    * 与文本内容相关的具体描述：
        - 占比、失败、未通过、问题数量、状态如何
    - 示例（DetailInfoTemplate）：
    ✅ "P08181 的项目总结是什么？" → DetailInfoTemplate（涉及 Project_Summary 文本内容）
    ✅ "P08181 的信息安全需求总数是多少？" → DetailInfoTemplate（涉及 CS_Status 文本内容）
    ✅ "P08181 的 Review 占比是多少？" → DetailInfoTemplate（涉及 CS_Status 文本内容）
    ✅ "P08181 的 JIRA 问题关闭率是多少？" → DetailInfoTemplate（涉及 Jira_Overview 文本内容）
    ✅ "P08181 的功能安全状况如何？" → DetailInfoTemplate（涉及 FS_Overview 文本内容）
    ✅ "P08181 的软件发布状态是什么？" → DetailInfoTemplate（涉及 SW_Release_Status 文本内容）
    ✅ "P08181 的 Tessy 测试结果有哪些问题？" → DetailInfoTemplate（涉及 Tessy_Detail 文本内容）
    ✅ "P08181 的 QAC 测试总结提到了什么？" → DetailInfoTemplate（涉及 QAC_Summary 文本内容）
    ✅ "P08181 的需求覆盖率是多少？" → DetailInfoTemplate（涉及 Base_On_Req1 文本内容）
    - 反例（NormalTemplate，而不是 DetailInfoTemplate）：
    ❌ "P08181 的项目编号和客户是什么？" → NormalTemplate（基本字段查询，不涉及大文本）
    ❌ "AE1 部门有哪些项目？" → NormalTemplate（列表查询，不涉及具体项目的文本字段）
    ❌ "2024年Q3有多少个新项目？" → NormalTemplate（简单统计查询，不涉及文本内容理解）
    ❌ "列出所有JAC客户的项目" → NormalTemplate（筛选查询，不涉及文本内容理解）
# 4. 风险相关分类（RiskRateTemplate / RiskDetermination）
    ## 4.1 RiskRateTemplate
    - 若问题是查询“具体的风险数据”，包括但不限于：
    * 风险评分（数值）
    * 高/中/低风险项目列表
    * 哪些项目风险最高
    * 显示风险评分大于 X 的项目
    则 classify it as 'RiskRateTemplate'。
    - 示例：
    - 查询高风险项目
    - 显示风险评分大于 0.8 的项目
    - 哪些项目风险最高
    ## 4.2 RiskDetermination
    - 若问题是询问“风险评定的方法、标准、流程、规则”等概念性内容，则：
    classify it as 'RiskDetermination'。
    - 示例：
    - 风险是如何评定的？
    - 风险评分标准是什么？
    - 风险等级划分规则？
# 5. 追问分类（FollowUpTemplate）—— 重点优化识别逻辑
    分析步骤（依次检查）：
    1. 检查是否存在指代词：
    - 这些、那些、这个、那个、该、此、它们、它、其中、当中等
    2. 检查时间指代：
    - 刚才、刚刚、上面、前面、之前、刚查询的等
    3. 检查结果指代：
    - 这些项目、那个项目、这些数据、上述结果、以上结果等
    4. 检查隐含追问表达：
    - 继续分析、进一步查看、详细信息、总结信息、更多信息、再详细一点、更具体一点等
    重要判断原则（满足即归为 FollowUpTemplate）：
    - 若问题包含 "这个项目"、"该项目"、"此项目" 等单数指代词，且同时满足：
    * 历史记录显示最近有“项目查询结果”（单个项目或明确对象），
    * 当前查询没有重新明确指定项目编号（如 P08379 等），
    * 上下文显示存在可被继续追问的数据，
    则 classify it as 'FollowUpTemplate'。
    - 若问题包含 "这些项目"、"那些项目" 等复数指代词，且：
    * 历史记录显示最近有“多项目查询结果”（如项目列表、统计结果等），
    则 classify it as 'FollowUpTemplate'。
    - 特别注意：
    即使问题文本本身看起来像“详细信息查询”（如 "总结信息"、"详细信息"），
    但如果：
    * 包含指代词（如：这个、这些、它、它们等），且
    * 有明确的上下文可指向（历史结果可对应），
    则优先 classify it as 'FollowUpTemplate'，
    而不是 DetailInfoTemplate。
    追问识别示例：
    ✅ "这个项目的总结信息" + 有历史项目查询 → FollowUpTemplate
    ✅ "这些项目中风险最高的" + 有历史项目列表 → FollowUpTemplate
    ✅ "其中哪个客户的项目最多" + 有历史统计结果 → FollowUpTemplate
    ✅ "它的详细信息" + 有历史单项目查询 → FollowUpTemplate
    ❌ "项目 P08379 的总结信息" + 明确项目号 → DetailInfoTemplate（不依赖上下文）
# 6. 继续查询分类（ContinueTemplate）
    - 若问题是非常明确的“继续当前展示或操作”的指令，而非新增语义查询，
    如：
    * "继续"
    * "下一页"
    * "往下看"
    * "接着上面的"
    * "再往后"
    等，则 classify it as 'ContinueTemplate'。

## 输出要求
你需要完成三件事：
1）输出完整的思维链推理过程（按步骤1-4清晰描述）；
2）为下面每一个类别计算一个 0 到 1 之间的置信度得分（score）。
   - 要求所有分数之和为 1。
   - 明显更符合的问题类别应当有更高的得分。
3）从中选出一个最合适的最终分类作为 "classify"，并生成符合格式要求的JSON。
   - "classify" 必须是得分最高的那个类别（如有并列最高，选你更确信的一个）。

可选类别列表（也是 scores 中必须出现的键）：
['HelpTemplate', 'Other_template', 'NormalTemplate', 'DetailInfoTemplate', 'RiskRateTemplate', 'RiskDetermination', 'FollowUpTemplate', 'ContinueTemplate']

### JSON输出格式要求
最终输出的JSON必须包含以下字段：
- "classify"：其值为以上类别中的一个。
- "scores"：一个对象，包含每个类别的得分。

JSON格式示例：
{{
  "classify": "<最终选择的类别名称>",
  "scores": {{
    "HelpTemplate": <0 到 1 的数字>,
    "Other_template": <0 到 1 的数字>,
    "NormalTemplate": <0 到 1 的数字>,
    "DetailInfoTemplate": <0 到 1 的数字>,
    "RiskRateTemplate": <0 到 1 的数字>,
    "RiskDetermination": <0 到 1 的数字>,
    "FollowUpTemplate": <0 到 1 的数字>,
    "ContinueTemplate": <0 到 1 的数字>
  }}
}}

### 额外要求
- "classify" 的值必须是上述类别之一。
- "scores" 中必须包含每一个类别的得分，且键名完全一致, 要求所有分数之和为 1。
- 建议保留 2 位小数（例如 0.85），但不是硬性要求。
- 输出中不要包含除 "classify" 和 "scores" 以外的其他字段。
- 先输出推理过程，再输出最终的JSON（仅JSON，无额外文本）。
"""


Human_prompt_text = """
请严格根据以下对话上下文，判断当前用户问题的分类类型：
对话上下文：
{history_context}

当前用户问题：
{question}
注意：
1. 上下文为空时，直接判断当前问题的独立类型；
2. 上下文非空时，优先检查是否存在追问特征（如“这个/这些/刚才”等指代词）；
3. 分类结果必须严格遵循给定的分类规则，优先匹配高优先级类别。
"""


class DynamicResponse(BaseModel):
    """
    LLM分类输出的结构化响应模型
    用于约束LLM输出的JSON格式，确保包含分类标签和各类别置信度得分
    """
    classify: str = Field(description="分类标签")
    scores: Dict[str, float] = Field(
        description = "每个类别的置信度得分，0-1之间，总和为1,且不必对某一个类别过于自信",
        # 分类失败模板
        default_factory= lambda: {
            "HelpTemplate": 0.0,
            "Other_template": 0.0,
            "NormalTemplate": 0.0,
            "DetailInfoTemplate": 0.0,
            "RiskRateTemplate": 0.0,
            "RiskDetermination": 0.0,
            "FollowUpTemplate": 0.0,
            "ContinueTemplate": 1.0
        }
    )


system_prompt = SystemMessagePromptTemplate.from_template(
    System_prompt_text_class_x,
    partial_variables={"format_instructions": JsonOutputParser(pydantic_object=DynamicResponse).get_format_instructions()
})
human_prompt = HumanMessagePromptTemplate.from_template(Human_prompt_text)
classification_template = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

SuppleInfo_prompt_text = """
下面是若干**已人工标注的问题及其正确意图类别**，请你把它们当作标准示例库，帮助你判断当前用户问题最合适的意图分类。

【已标注示例（JSON 数组或对象形式，每条通常包含 question 与 label 等字段）】
{suppleInfoList}

【当前待分类问题】
{question}

你需要完成的任务：
1. 结合上面的人工标注示例，比较当前问题与各示例在语义上的相似度；
2. 只允许在这些示例中出现过的意图类别（即示例中的 label/intent 字段取值）中选择一个，作为当前问题的最终分类；
3. 不得创建任何示例库中未出现的新类别名称。

输出要求：
1. 只输出一个 JSON 对象，不要输出任何思维链、解释或额外文本；
2. JSON 对象必须严格包含以下两个字段：
   - "classify": 字符串，值必须为示例库中出现过的某个意图类别名称（例如各条数据中的 label 字段取值）；
   - "scores": 对象，键为示例库中出现过的所有意图类别名称，值为该类别对当前问题的置信度（0~1 之间的数字，所有得分之和必须为 1）。
3. "classify" 必须是得分最高的那个类别（如有并列，选择你最确信的一个）。
"""

classification_second_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(SuppleInfo_prompt_text)
])

def truncate_column_with_ellipsis(df, column_name, max_length=200):
    """截取文本并在超长时添加省略号"""
    df_result = df.copy()
    
    def truncate_text(text):
        if pd.isna(text):
            return text
        
        text = str(text)
        if len(text) > max_length:
            return text[:max_length-3] + '...'
        return text
    
    df_result[column_name] = df_result[column_name].apply(truncate_text)
    return df_result

def truncate_long_text_fields(df, max_length=2000):
    """
    统一截断DataFrame中的所有长文本字段
    
    Args:
        df: 输入DataFrame
        max_length: 截断长度（默认2000字符）
        
    Returns:
        df_result: 截断后的DataFrame
    """
    import pandas as pd
    if df is None or df.empty:
        return df
    
    df_result = df.copy()
    
    # 只处理字符串类型的列
    for column in df_result.columns:
        if df_result[column].dtype == 'object':
            # 检查是否为长文本字段
            if column in Config.LONG_TEXT_COLUMNS:
                df_result[column] = df_result[column].apply(
                    lambda x: str(x)[:max_length-3] + '...' if pd.notna(x) and len(str(x)) > max_length else x
                )
    
    return df_result

class Pipeline:
                       
    def __init__(self,max_history_length: int = 10):
        self.name = "Project AI 3.3"

        self.user_logger = UserQueryLogger()  # 使用Config默认配置

        self.df_chunks = None
        self.current_chunk = 0
        self.chunk_size = 0

        # 添加对话管理器（简化版）
        self.conversation_manager =  ConversationManager(
            max_history_length=max_history_length
        )

        # 使用Xinference替换ollama
        print("正在初始化Xinference客户端...")
        # 一阶主模型：用于常规意图识别
        self.llm = ChatOpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
        )
        # 二阶小模型：仅在不确定时触发的二次咨询，通常配置为更小、更快的模型
        self.llm_second = ChatOpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            model=Config.LLM_SECOND_MODEL,
            temperature=Config.LLM_TEMPERATURE,
        )
        print("Xinference客户端初始化完成")
        
        # 指标收集
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_call_times = []
        self.llm_call_count = 0
        self.e2e_times = []
        self.e2e_count = 0

    def force_clean_think_tags(self, text):
        """强制移除所有think相关内容"""
        patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'</?think\b[^>]*>',
            r'</?thinking\b[^>]*>',
            r'\[think\].*?\[/think\]',
            r'\[thinking\].*?\[/thinking\]',

        ]

        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # 提取第一个df表达式
        lines = cleaned.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('df[') or line.startswith('df.') or line.startswith('last_df[') or line.startswith('last_df.'):
                # 关键修复：修复pandas布尔运算和优先级问题
                line = self.fix_pandas_boolean_emergency(line)
                return line
        
        # 如果没有找到df表达式，返回清理后的文本
        return cleaned.strip()

    def clean_think_tags_from_answer(self, text):
        """从文本答案中移除think标签（不提取df表达式）"""
        patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'</?think\b[^>]*>',
            r'</?thinking\b[^>]*>',
            r'\[think\].*?\[/think\]',
            r'\[thinking\].*?\[/thinking\]',
        ]

        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # 清理多余的空白行
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()

    def track_llm_call(self, chain, inputs, description=""):
        """包装LLM调用以跟踪token使用和延迟"""
        start_time = time.time()
        result = chain.invoke(inputs)
        # print('track_llm_call_result', result)
        call_time = time.time() - start_time
        
        self.llm_call_times.append(call_time)
        self.llm_call_count += 1
        
        # 尝试获取token使用信息
        try:
            # LangChain可能返回response_metadata
            if hasattr(result, 'response_metadata'):
                metadata = result.response_metadata
                if 'token_usage' in metadata:
                    token_usage = metadata['token_usage']
                    self.total_tokens += token_usage.get('total_tokens', 0)
                    self.prompt_tokens += token_usage.get('prompt_tokens', 0)
                    self.completion_tokens += token_usage.get('completion_tokens', 0)
            # 如果没有token信息，至少记录时间
            if description:
                print(f"[{description}] 耗时: {call_time:.3f}s")
        except Exception as e:
            if description:
                print(f"[{description}] 耗时: {call_time:.3f}s (token信息不可用)")
        
        return result


    def extract_full_rows(self, original_df, result_df, generated_code):
        """
        从原始DataFrame中提取完整行数据，用于后续追问
        
        Args:
            original_df: 原始完整数据
            result_df: 查询结果（可能只包含部分列）
            generated_code: 生成的查询代码
        
        Returns:
            DataFrame: 包含完整列的结果数据
        """
        try:
            # 如果结果为空，返回空的完整DataFrame
            if result_df.empty:
                return original_df.iloc[0:0].copy()  # 返回空的但包含所有列的DataFrame
            
            # 尝试通过ProjectNo匹配获取完整行（最可靠的方法）
            if 'ProjectNo' in result_df.columns and 'ProjectNo' in original_df.columns:
                project_nos = result_df['ProjectNo'].tolist()
                full_rows = original_df[original_df['ProjectNo'].isin(project_nos)]
                if len(full_rows) == len(result_df):
                    print(f"通过ProjectNo成功匹配到{len(full_rows)}行完整数据")
                    return full_rows.copy()
            
            # 备选方案1：尝试通过索引匹配
            if hasattr(result_df, 'index') and len(result_df) <= len(original_df):
                try:
                    # 如果索引仍然有效，直接使用
                    if all(idx in original_df.index for idx in result_df.index):
                        full_rows = original_df.loc[result_df.index]
                        print(f"通过索引成功匹配到{len(full_rows)}行完整数据")
                        return full_rows.copy()
                except Exception as e:
                    print(f"索引匹配失败: {e}")
            
            # 备选方案2：如果是单行结果，尝试多列匹配
            if len(result_df) == 1:
                # 使用多个关键列进行匹配
                match_columns = ['Department', 'Customer', 'ProductVariant']
                available_match_cols = [col for col in match_columns if col in result_df.columns and col in original_df.columns]
                
                if available_match_cols:
                    conditions = []
                    for col in available_match_cols:
                        value = result_df.iloc[0][col]
                        if pd.notna(value):
                            conditions.append(original_df[col] == value)
                    
                    if conditions:
                        # 使用 & 连接所有条件
                        combined_condition = conditions[0]
                        for condition in conditions[1:]:
                            combined_condition = combined_condition & condition
                        
                        matched_rows = original_df[combined_condition]
                        if len(matched_rows) >= 1:
                            print(f"通过多列匹配到{len(matched_rows)}行完整数据")
                            return matched_rows.head(1).copy()  # 如果多行匹配，取第一行
            
            # 最后备选：返回原始结果（虽然列不完整，但至少可用）
            print("无法获取完整行数据，使用原始查询结果")
            return result_df.copy()
            
        except Exception as e:
            print(f"提取完整行数据时出错: {e}")
            # 出错时返回原始结果
            return result_df.copy()


    def fix_pandas_boolean_emergency(self, code):
        """紧急修复pandas布尔运算，包括优先级问题"""
        import re
        
        original_code = code
        # print(f"原始代码: {original_code}")
        
        # 步骤1：替换 and/or 为 &/|
        code = re.sub(r'\band\b', '&', code)
        code = re.sub(r'\bor\b', '|', code)
        
        # 步骤2：处理运算符优先级问题 - 为df[...]内的条件添加括号
        def fix_df_brackets(match):
            df_name = match.group(1)  # df 或 last_df
            condition = match.group(2)
            
            # 如果条件包含 & 或 |，需要确保每个比较操作都有括号
            if '&' in condition or '|' in condition:
                # 分割条件，保留操作符
                parts = re.split(r'(\s*[&|]\s*)', condition)
                
                fixed_parts = []
                for i, part in enumerate(parts):
                    if i % 2 == 0:  # 比较条件部分
                        part = part.strip()
                        if part and not (part.startswith('(') and part.endswith(')')):
                            # 只对包含比较操作符的部分添加括号
                            if any(op in part for op in ['>', '<', '==', '!=', '>=', '<=']):
                                fixed_parts.append(f"({part})")
                            else:
                                fixed_parts.append(part)
                        else:
                            fixed_parts.append(part)
                    else:  # & 或 | 操作符
                        fixed_parts.append(part)
                
                condition = ''.join(fixed_parts)
            
            return f"{df_name}[{condition}]"
        
        # 应用修复 - 支持 df 和 last_df
        code = re.sub(r'((?:last_)?df)\[([^\[\]]+)\]', fix_df_brackets, code)
        
        # 步骤3：特殊处理常见的错误模式
        error_patterns = [
            # 修复形如：df[condition1 & condition2] 没有括号的情况
            (r"((?:last_)?df)\[([^()\[\]]+\s*[><=!]+\s*[^&|()]+)\s*([&|])\s*([^()\[\]]+\s*[><=!]+\s*[^&|()]+)\]",
             r"\1[(\2) \3 (\4)]"),
        ]
        
        for pattern, replacement in error_patterns:
            code = re.sub(pattern, replacement, code)
        
        # 步骤4：清理多余空格
        code = re.sub(r'\s+', ' ', code).strip()
        
        # print(f"修复后代码: {code}")
        return code

    def callAgent(self, df, query, prompt, use_last_result=False):
        """执行代理查询，支持基于上次结果的查询"""
        self.df = df
        self.max_retries = Config.MAX_RETRIES
        
        # 检测用户是否明确查询Unknown或空值
        query_lower = query.lower()
        # 检查是否包含"unknown"、"空"、"nan"、"none"等关键词
        unknown_keywords = ['unknown', '是unknown', '为unknown', '空', '为空', '是空', 'nan', 'none']
        is_querying_unknown_or_empty = any(keyword in query_lower for keyword in unknown_keywords)
        
        # 设置执行环境
        if use_last_result and self.conversation_manager.has_previous_result_full():
            # 使用上次结果的完整版本作为last_df
            self.tool = PythonAstREPLTool(locals={
                "df": df, 
                "last_df": self.conversation_manager.last_query_result_full
            })
            available_columns = list(self.conversation_manager.last_query_result_full.columns)
        else:
            self.tool = PythonAstREPLTool(locals={"df": df})
            available_columns = list(df.columns)
        
        self.allname = ','.join(df.columns)
        
        # 构建模型链
        self.chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        
        i = 0
        result = None
        finalresult = None
        full_result = None  # 新增：保存完整结果
        seen_errors = set()  # 记录已见过的错误，避免重复
        pandas_error_count = 0  # 记录pandas错误次数
        
        while i < self.max_retries:
            try:
                # 生成代码
                error_context = ""
                if result is not None:
                    print(f"上次发生错误: {result}")
                    error_context = str(result)
                    
                    # 如果是重复错误或连续pandas错误，直接退出
                    if error_context in seen_errors:
                        print("检测到重复错误，停止重试")
                        return "抱歉，查询执行失败。请尝试重新描述您的问题。", None, None
                    
                    if "truth value of a Series is ambiguous" in error_context:
                        pandas_error_count += 1
                        if pandas_error_count >= 3:
                            print("连续出现pandas错误，停止重试")
                            return "查询失败：请尝试更简单的查询，如'查询AE1部门的项目'", None, None
                    
                    seen_errors.add(error_context)

                if use_last_result:
                    # 生成追问查询的代码
                    code = self.track_llm_call(
                        self.chain,
                        {
                            "question": query,
                            "context_info": "使用 last_df 进行追问查询",
                            "available_columns": ', '.join(available_columns),
                            "lastcode": result
                        },
                        description="代码生成(追问)"
                    )
                else:
                    code = self.track_llm_call(
                        self.chain,
                        {
                            "question": query, 
                            "Columnname": self.allname, 
                            "lastcode": result
                        },
                        description="代码生成"
                    )
                    
                code = self.force_clean_think_tags(code)
                print(f"生成的代码: {code}")  # Debug用

                # 执行代码
                try:
                    result = self.tool.run(code)
                    pass
                except Exception as e:
                    error_msg = str(e)
                    print(f"执行错误: {error_msg}")
                    result = error_msg
                    i += 1
                    continue

                if isinstance(result, int):
                    print(f"警告：返回类型不是DataFrame，而是{type(result)}")
                    self.df_chunks = None
                    self.current_chunk = 0
                    self.chunk_size = 0
                    self.df_raw_total = 0
                    finalresult = result
                    full_result = result
                    break
                elif isinstance(result, pd.DataFrame):
                    # 只有当用户不是明确查询Unknown或空值时，才过滤掉包含"Unknown"或空值的记录
                    if not is_querying_unknown_or_empty:
                        def filter_unknown_and_empty(df):
                            """过滤掉包含空值的记录，但保留-999（表示Unknown的整数标记值）"""
                            mask = True
                            for col in df.columns:
                                # 检查是否为字符串类型的"Unknown"（不区分大小写），但排除-999
                                is_unknown = (df[col].astype(str).str.lower() == 'unknown') & (df[col] != Config.UNKNOWN_INT_VALUE)
                                # 检查是否为空值（NaN、None、空字符串等），但排除-999（Unknown标记值）
                                is_empty = df[col].isna() | (df[col].astype(str).str.strip() == '')
                                # 排除-999（Unknown标记值），不将其视为空值
                                is_empty = is_empty & (df[col] != Config.UNKNOWN_INT_VALUE)
                                # 如果该列包含Unknown或空值，则该行被过滤
                                mask = mask & (~is_unknown) & (~is_empty)
                            return df[mask]
                        # 应用过滤
                        result = filter_unknown_and_empty(result)                  
                    if len(result) == 0:
                        print("查询结果为空DataFrame（过滤Unknown和空值后）")
                        self.df_chunks = None
                        self.current_chunk = 0
                        self.chunk_size = 0
                        self.df_raw_total = 0
                        finalresult = "未找到符合条件的项目数据，请尝试调整查询条件。"  # 返回提示信息
                        full_result = result.copy()
                        break
                    # 根据结果列决定chunk size
                    # 如果包含非结构化数据列，使用更小的chunk size
                    result_columns = set(result.columns)
                    non_structured_columns = set(Config.LONG_TEXT_COLUMNS)
                    has_non_structured_data = len(result_columns.intersection(non_structured_columns)) > 0                  
                    if has_non_structured_data:
                        self.chunk_size = 5  # 非结构化数据列最多返回5条/块
                    else:
                        self.chunk_size = Config.CHUNK_SIZE  # 其他情况使用默认值                    
                    # 截断所有长文本字段
                    result_truncated = truncate_long_text_fields(result, Config.LONG_TEXT_MAX_LENGTH)
                    
                    self.df_chunks = [result_truncated.iloc[i:i+self.chunk_size] for i in range(0, len(result_truncated), self.chunk_size)]
                    self.current_chunk = 0
                    self.df_raw_total = len(result_truncated)
                    finalresult = self.df_chunks[self.current_chunk]
                    
                    # 重要：为追问准备完整数据
                    if not use_last_result:
                        # 如果是主查询，需要获取完整行数据以备追问
                        full_result = self.extract_full_rows(df, result, code)
                    else:
                        # 如果是追问，使用当前结果作为完整结果
                        full_result = result.copy()
                    break
                elif isinstance(result, np.ndarray):
                    print(f"警告：返回类型不是DataFrame，而是{type(result)}")
                    self.df_chunks = None
                    self.current_chunk = 0
                    self.chunk_size = 0
                    self.df_raw_total = 0
                    finalresult = result
                    full_result = result
                    break
                elif isinstance(result,pd.Series):
                    print(f"警告：返回类型不是DataFrame，而是{type(result)}")
                    # 修复：将Series转换为DataFrame，保留所有行
                    result_df = result.to_frame()
                    self.chunk_size = Config.CHUNK_SIZE
                    self.df_chunks = [result_df.iloc[i:i+self.chunk_size] for i in range(0, len(result_df), self.chunk_size)]
                    self.current_chunk = 0
                    self.df_raw_total = len(result_df)
                    finalresult = self.df_chunks[self.current_chunk]
                    
                    # 为Series结果也尝试获取完整数据
                    if not use_last_result:
                        full_result = self.extract_full_rows(df, result_df, code)
                    else:
                        full_result = result_df.copy()
                    break
                else:
                    i = i + 1
            except Exception as e:
                i = i + 1
                error_msg = str(e)
                print(f"生成代码时出错: {error_msg}")
                result = error_msg
        
        # 如果所有重试都失败，返回友好的错误信息
        if finalresult is None:
            return "抱歉，无法执行您的查询。请尝试使用更简单的描述，比如：'查询AE1部门的项目'或'显示项目P08379的信息'", None, None
        
        return finalresult, code, full_result  # 返回结果、生成的代码和完整结果

    def answer_from_text_field(self, question: str, project_no: str, field_name: str) -> str:
        """
        从大文本字段中提取信息回答问题

        Args:
            question: 用户问题
            project_no: 项目编号
            field_name: 文本字段名（如 'CS_Status', 'Jira_Overview'等）

        Returns:
            str: 自然语言答案
        """
        try:
            # 获取完整数据
            df = get_products_df()

            # 定位项目记录
            project_records = df[df['ProjectNo'] == project_no]

            if len(project_records) == 0:
                return f"未找到项目 {project_no} 的记录"

            num_records = len(project_records)

            # 如果有多条记录，分别处理每条记录
            if num_records > 1:
                print(f"[成功] 项目{project_no}有{num_records}条记录，分别分析")

                answers = []
                understanding_chain = text_understanding_template | self.llm | StrOutputParser()

                for idx, (_, record) in enumerate(project_records.iterrows(), 1):
                    # 获取记录的标识信息（使用业务字段）
                    project_no_val = record.get('ProjectNo', '')
                    customer = record.get('Customer', '')
                    carline = record.get('Carline', '')
                    product_variant = record.get('ProductVariant', '')

                    # 构建记录标识
                    record_id_parts = []
                    if pd.notna(project_no_val) and project_no_val:
                        record_id_parts.append(f"项目: {project_no_val}")
                    if pd.notna(customer) and customer:
                        record_id_parts.append(f"客户: {customer}")
                    if pd.notna(carline) and carline:
                        record_id_parts.append(f"车型: {carline}")
                    if pd.notna(product_variant) and product_variant:
                        record_id_parts.append(f"产品: {product_variant}")

                    record_id = f"记录{idx} ({', '.join(record_id_parts)})" if record_id_parts else f"记录{idx}"

                    # 获取文本字段内容
                    text_content = record[field_name]

                    if pd.isna(text_content) or text_content == '':
                        answers.append(f"**{record_id}**: {field_name} 字段为空")
                        continue

                    # 截断过长的文本
                    max_text_length = Config.LONG_TEXT_MAX_LENGTH
                    if len(text_content) > max_text_length:
                        text_content = text_content[:max_text_length] + "\n\n(内容过长，已截断)"

                    # 使用LLM理解文本并回答
                    answer = self.track_llm_call(
                        understanding_chain,
                        {
                            "project_no": project_no_val,
                            "field_name": field_name,
                            "text_content": text_content,
                            "question": question
                        },
                        description=f"文本理解(记录{idx}/{num_records})"
                    )

                    # 去掉 <think></think> 标签
                    answer = self.clean_think_tags_from_answer(answer)

                    answers.append(f"**{record_id}**:\n{answer.strip()}")

                # 合并所有答案
                final_answer = f"项目 {project_no} 有 {num_records} 条记录，分析如下：\n\n" + "\n\n---\n\n".join(answers)
                return final_answer

            else:
                # 只有一条记录，使用原逻辑
                print(f"[成功] 项目{project_no}有1条记录")

                # 获取文本字段内容
                text_content = project_records.iloc[0][field_name]

                if pd.isna(text_content) or text_content == '':
                    return f"项目 {project_no} 的 {field_name} 字段为空"

                # 截断过长的文本（避免token超限）
                max_text_length = Config.LONG_TEXT_MAX_LENGTH
                if len(text_content) > max_text_length:
                    text_content = text_content[:max_text_length] + "\\n\\n(内容过长，已截断)"
                    print(f"[警告] 文本内容过长，已截断至{max_text_length}字符")

                # 使用LLM理解文本并回答
                understanding_chain = text_understanding_template | self.llm | StrOutputParser()
                answer = self.track_llm_call(
                    understanding_chain,
                    {
                        "project_no": project_no,
                        "field_name": field_name,
                        "text_content": text_content,
                        "question": question
                    },
                    description="文本理解"
                )

                # 去掉 <think></think> 标签
                answer = self.clean_think_tags_from_answer(answer)

                return answer.strip()

        except Exception as e:
            print(f"[失败] 文本字段理解失败: {e}")
            import traceback
            traceback.print_exc()
            return f"处理文本字段时出错: {str(e)}"


    def detect_and_handle_text_field_query(self, question: str, messages: list, from_followup: bool = False):
        """
        检测并处理文本字段查询

        Args:
            question: 用户问题
            messages: Open-WebUI 的消息历史
            from_followup: 是否来自追问（会从历史记录中提取项目编号）

        Returns:
            str or None: 如果是文本字段查询，返回答案；否则返回None
        """
        import re

        # 定义文本字段关键词映射（使用更精确的关键词）
        text_field_keywords = {
            'Project_Summary': [
                '项目总结', '项目概览', '项目概述', '整体状况','项目概况',
                '详细信息', '详情', '总结', '项目详情', '项目信息', '项目情况'
            ],
            'CS_Status': [
                '信息安全', '网络安全', 'CS状态','CS Overview总结','信息安全需求','测试失败的需求','Review占比','CS测试情况'
            ],
            'Jira_Overview': [
                'JIRA', 'Jira', 'jira', 'JIRA问题', 'JIRA概览','总体JIRA概况', '客户端JIRA概况', 'JIRA Problem Found By分布', 'JIRA Change变更类型分布', '风险项'
            ],
            'FS_Overview': [
                '功能安全', 'FS概览', 'FS状态',' 功能安全需求总数量',' 功能安全需求','FS Overview总结 ','FS总结'
            ],
            'Base_On_Req1': [
               '系统需求覆盖率', '系统需求','客户需求',' 系统架构设计（System Architecture Design）',' 系统架构设计','软件需求','硬件需求','机械需求','风险项识别'
            ],
            'Base_On_Req2': [
                '需求管理', '需求分析','Test Overview详情',' Test Overview总结','测试未覆盖的需求','测试结果不全的需求','测试失败的需求'
            ],
            'SW_Release_Status': [
                '软件发布状态', '软件释放情况','SW_Release_Status总结',' Sample A','Sample B','Sample C','风险项'
            ],
            'Tessy_Summary': [
                'Tessy总结', 'Tessy概览','测试概况','测试总结','测试概览','测试概况总结','测试概述','Tessy测试结果','Tessy测试','Tessy'
            ],
            'Base_On_TestCase': [
                '测试用例', '测试用例分析',' BaseOnTestcase总结','HIL测试','HILP测试','HWT测试',' ITSW测试','UTSW测试','MECH测试','风险项'
            ],
            'QAC_Summary': [
                'QAC测试总结', 'QAC测试概况','QAC静态代码检查报告总结','QAC','静态代码检查','静态代码'
            ],

            'CS_Status_Detail': [
                'CS详细状态', '信息安全详情','CS测试情况'
            ],
            'Jira_Detail': [
                'JIRA详情', 'Jira详细信息'
            ],
            'Tessy_Detail': [
                'Tessy详情'
            ],
            'FS_Detail': [
                '功能安全详情', 'FS详细信息'
            ],
            'QAC_Detail': [
                'QAC详情'
            ],

        }

        # 尝试提取项目编号
        project_pattern = r'P\d{5}(?:\.\d+)?'
        project_match = re.search(project_pattern, question)

        project_no = None

        if project_match:
            project_no = project_match.group(0)
            print(f"[成功] 从问题中检测到项目编号: {project_no}")
        elif from_followup:
            # 从 messages 中提取项目编号，如果messages为空，则使用conversation_manager.history
            history_to_use = messages if messages else self.conversation_manager.history
            project_no = ConversationManager.extract_project_from_messages(history_to_use)
            if project_no:
                print(f"[成功] 从历史消息中提取到项目编号: {project_no}")
            else:
                print("[警告] 追问查询但无法从历史消息中提取项目编号")
                return None
        else:
            print("[警告] 未检测到项目编号，不是文本字段查询")
            return None

        # 检测涉及哪个文本字段
        detected_field = None
        detected_keyword = None
        max_keyword_length = 0
        
        # 优先匹配更长、更精确的关键词
        for field_name, keywords in text_field_keywords.items():
            for keyword in keywords:
                if keyword in question:
                    # 选择最长的匹配关键词
                    if len(keyword) > max_keyword_length:
                        max_keyword_length = len(keyword)
                        detected_field = field_name
                        detected_keyword = keyword
        
        if detected_field:
            print(f"[成功] 检测到文本字段: {detected_field} (关键词: {detected_keyword})")

        # 如果未检测到字段，且是追问，尝试从消息历史中获取上一次的字段
        if not detected_field and from_followup:
            # 使用与提取项目编号相同的历史消息源
            history_to_use = messages if messages else self.conversation_manager.history
            detected_field = ConversationManager.extract_last_text_field_from_messages(history_to_use)
            if detected_field:
                print(f"[成功] 从历史消息中推断字段: {detected_field}")
            else:
                print("[警告] 无法从历史消息推断字段，使用常规流程")
                return None
        elif not detected_field:
            print("[警告] 未检测到文本字段关键词，使用常规流程")
            return None

        # 调用文本字段理解函数
        print(f"[调试] 调用文本字段理解: project={project_no}, field={detected_field}")
        answer = self.answer_from_text_field(question, project_no, detected_field)

        return answer

    def check_need_second_agent(self, result, min_threshold: float = 0.8, diff_threshold: float = 0.2):
        """
        根据分类结果的 scores 判断是否需要启动第二代理。
        当 scores 缺失或为空时，默认不触发第二代理。
        """
        scores = None
        if isinstance(result, dict):
            scores = result.get("scores")
        if not scores:
            return {"need_consult_agent": 0}
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_class, top_score = sorted_items[0]
        second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        cond_a = top_score <= min_threshold
        cond_b = (top_score - second_score) < diff_threshold
        need_consult_agent = 1 if (cond_a or cond_b) else 0
        return {"need_consult_agent": need_consult_agent}

    def rule_based_classify(self, question: str) -> str | None:
        """
        简单规则优先分类：
        - 明显的帮助类问题 -> HelpTemplate
        - 明确的继续类指令（且有上一轮结果） -> ContinueTemplate
        - 明显与项目管理无关的问题 -> Other_template
        命中规则时直接返回类别名称，否则返回 None 交给 LLM 处理。
        """
        text = (question or "").strip()
        if not text:
            return None

        # 1) 帮助类：典型“你能做什么/帮我/使用说明”之类，且文本较短
        help_keywords = ["帮助", "使用说明", "功能介绍", "你能做什么", "如何使用"]
        if len(text) <= 15 and any(kw in text for kw in help_keywords):
            return "HelpTemplate"

        # 2) 继续类：只有一个简单指令词，且有上一轮查询结果可继续
        continue_keywords = {"继续", "下一页", "往下看", "接着上面的", "再往后"}
        if text in continue_keywords and self.conversation_manager.has_previous_result_full():
            return "ContinueTemplate"

        return None

    # 意图识别分类器
    def prompt_router(self, query, messages, suppleInfo=None):
        """路由查询到合适的处理器，支持多轮对话

        Args:
            query: 包含用户输入的字典 {'input': '...'}
            messages: Open-WebUI 的消息历史列表
        """

        # 规则优先：先用轻量规则拦截明显样本，命中则直接返回
        rule_class = self.rule_based_classify(query.get("input", ""))
        if rule_class is not None:
            # 构造与 DynamicResponse 兼容的简单 scores
            base_scores = {
                "HelpTemplate": 0.0,
                "Other_template": 0.0,
                "NormalTemplate": 0.0,
                "DetailInfoTemplate": 0.0,
                "RiskRateTemplate": 0.0,
                "RiskDetermination": 0.0,
                "FollowUpTemplate": 0.0,
                "ContinueTemplate": 0.0,
            }
            if rule_class in base_scores:
                base_scores[rule_class] = 1.0
            return {
                "classify": rule_class,
                "scores": base_scores,
            }

        # 从 messages 中提取历史上下文
        history_context = ConversationManager.extract_context_from_messages(messages)
        # print(f"调试信息:")
        # print(f"   查询: {query['input']}")
        # print(f"   消息历史数: {len(messages)}")
        # print(f"   有上次结果: {self.conversation_manager.has_previous_result_full()}")
        # print(f"   历史上下文: {history_context}")

        classification_chain = classification_template | self.llm
        # 先生成文本，然后清理并手动解析JSON
        classification_result = self.track_llm_call(
            classification_chain,
            {
                "question": query["input"],
                "history_context": history_context
            },
            description="分类"
        )
        
        # 提取AIMessage的内容文本
        if hasattr(classification_result, "content"):
            classification_text = classification_result.content
        else:
            classification_text = str(classification_result)

        # 清理可能的</think>标签和其他非JSON内容
        classification_text = self.clean_think_tags_from_answer(classification_text)

        def _parse_classification_json(text: str) -> dict:
            """将模型输出文本稳健地解析为 JSON dict，失败时返回默认分类。"""
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                try:
                    json_start = text.find("{")
                    json_end = text.rfind("}") + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = text[json_start:json_end]
                        return json.loads(json_str)
                    else:
                        # 只打印前 200 个字符避免控制台被长文本刷屏
                        preview = (text[:200] + "...") if len(text) > 200 else text
                        print(f"无法从输出中提取JSON，输出预览: {preview}")
                        return {"classify": "NormalTemplate"}
                except Exception:
                    preview = (text[:200] + "...") if len(text) > 200 else text
                    print(f"JSON解析失败，输出预览: {preview}")
                    return {"classify": "NormalTemplate"}

        # 首次分类结果解析
        classification = _parse_classification_json(classification_text)

        classificationisOk = False

        # 二次咨询触发：当一阶模型不自信时，使用更小、更快的模型做二次判定
        need_second = self.check_need_second_agent(classification).get("need_consult_agent", 0)
        if need_second:
            # 只保留与一阶 scores 中「置信度大于 0」的类别相关的 suppleInfo 子集
            scores = classification.get("scores") or {}
            non_zero_labels = {label for label, score in scores.items() if score and score > 0}

            supple_subset = suppleInfo
            if isinstance(suppleInfo, list) and non_zero_labels:
                filtered = [
                    item
                    for item in suppleInfo
                    if isinstance(item, dict) and item.get("label") in non_zero_labels
                ]
                # 如果过滤后不为空，则使用子集；否则退回原始 suppleInfo，避免误删
                if filtered:
                    supple_subset = filtered

            classification_second_chain = classification_second_template | self.llm_second
            classification_seconda_result = self.track_llm_call(
                classification_second_chain,
                {
                    "question": query["input"],
                    "suppleInfoList": supple_subset,
                },
                description="二次分类",
            )
            # 提取AIMessage的内容文本
            if hasattr(classification_seconda_result, "content"):
                second_classification_text = classification_seconda_result.content
            else:
                second_classification_text = str(classification_seconda_result)

            # 清理可能的</think>标签和其他非JSON内容
            second_classification_text = self.clean_think_tags_from_answer(second_classification_text)

            # 使用同一解析逻辑解析二次分类结果
            result_second_classification = _parse_classification_json(second_classification_text)
            return result_second_classification

        return classification
        
        # 判断classification是否是字典
        if isinstance(classification, dict):
            # 判断是否含有键'classify'
            if 'classify' in classification:
                print("classification是字典，并且含有键'classify'")
                classificationisOk = True
            else:
                print("不含有键'classify'")
        else:
            print("classification不是字典")

        if not classificationisOk:
            return "分类失败，请重试！"
        
        classify = classification["classify"]
        result = None
        generated_code = None
        
        # 特殊处理：当用户询问"提供数据"、"有哪些数据"等时，固定返回指定列
        data_inquiry_keywords = [
            "提供哪些项目相关的数据",
            "提供哪些数据",
            "有哪些项目相关的数据",
            "有哪些数据",
            "可以提供哪些数据",
            "可以查询哪些数据",
            "能提供哪些数据",
            "能查询哪些数据",
            "你可以提供哪些项目相关的数据",
            "你可以提供哪些数据"
        ]
        
        # 检查是否为数据询问类问题
        is_data_inquiry = any(keyword in query["input"] for keyword in data_inquiry_keywords)
        
        if is_data_inquiry:
            # 固定返回指定的5列数据
            df = get_products_df()
            result = df[['Department', 'ProjectNo', 'Customer', 'ProductVariant', 'Carline']]
            generated_code = "df[['Department','ProjectNo', 'Customer', 'ProductVariant', 'Carline']]"
            full_result = result
            
            # 保存查询结果用于追问
            self.conversation_manager.save_query_result(
                result=result,
                full_result=full_result,
                is_followup=False
            )
            
            # 保存对话历史
            self.conversation_manager.add_message("user", query["input"])
            result_str = str(result) if result is not None else "无结果"
            self.conversation_manager.add_message("assistant", result_str)
            
            return result
        
        try:
            if "HelpTemplate" in classify:
                user_question = query["input"].lower()
                
                # 定义不同问题的回复内容
                if any(question in user_question for question in ["你是谁", "自我介绍", "认识你", "什么是你", "你是什么","你好"]):
                    help_text = "👋 你好！我是KOSTAL项目管理智能助手，专为KOSTAL项目团队设计，可通过自然语言查询项目数据，助力项目决策与风险管控。"
                    return help_text
                
                elif any(question in user_question for question in ["你可以干什么", "能做什么", "功能", "能力", "用途","你可以干啥"]):
                    help_text = """🎯 我主要提供以下功能：
1. 项目查询 - 查询项目基本信息、客户、产品变体、车型等
2. 详细信息 - 查询项目总结、CS状态、JIRA问题、功能安全等详细信息
3. 风险分析 - 查询项目风险评分、高风险项目列表
4. 统计分析 - 按部门、客户、样件阶段等进行统计分析
"""
                    return help_text
                
                elif any(question in user_question for question in ["查询示例", "例子", "示例", "举个例子", "示范"]):
                    help_text = """💡 查询示例：
【基本信息查询】
• "AE1部门有哪些项目？"
• "查询项目P08379的基本信息"
• "显示P09003的信息"
• "客户是JAC的项目有哪些？"

【详细信息查询】
• "项目P08379的详细信息"
• "P08181的项目总结是什么？"
• "P08181的JIRA问题关闭率是多少？"
• "P08181的功能安全状况如何？"

【风险分析查询】
• "AE1部门的高风险项目有哪些？"
• "显示风险评分大于0.8的项目"
• "哪些项目风险最高？"
• "AE2部门的中风险项目"

【数据筛选查询】
• "测试通过率大于80%的项目"
• "A样件阶段的所有项目"
• "查询客户名称包含'JAC'的项目"
• "风险评分大于0.8且客户是JAC的项目"
"""
                    return help_text
                
                elif any(question in user_question for question in ["使用说明", "说明", "帮助", "如何使用", "用法"]):
                    help_text = """- 使用说明：
我是基于自然语言处理技术的智能助手，您可以通过中文自然语言向我提问，查询KOSTAL项目相关数据。

🔧 支持的操作：
• 筛选：根据部门、客户、风险等级等条件筛选
• 排序：按风险评分、测试通过率等字段排序
• 追问：基于上次查询结果进行进一步分析
"""
                    return help_text
                
                elif any(question in user_question for question in ["使用技巧", "技巧", "提示", "建议", "注意事项"]):
                    help_text = """📝 使用技巧：
1. 使用自然语言描述您的需求
2. 可以指定项目编号（如P08379）进行精确查询
3. 可以组合多个条件进行复杂查询
4. 支持中文查询，不区分大小写
5. 可以基于上次查询结果进行追问
"""
                    return help_text
                
                else:
                    # 默认回复完整的帮助信息
                    help_text = """👋 你好！我是KOSTAL项目管理智能助手，可通过自然语言查询项目数据。 
- 使用说明
🎯 主要功能：
1. 项目查询 - 查询项目基本信息、客户、产品变体、车型等
2. 详细信息 - 查询项目总结、CS状态、JIRA问题、功能安全等详细信息
3. 风险分析 - 查询项目风险评分、高风险项目列表
4. 统计分析 - 按部门、客户、样件阶段等进行统计分析

💡 查询示例：
【基本信息查询】
• "AE1部门有哪些项目？"
• "查询项目P08379的基本信息"
• "显示P09003的信息"
• "客户是JAC的项目有哪些？"

【详细信息查询】
• "项目P08379的详细信息"
• "P08181的项目总结是什么？"
• "P08181的JIRA问题关闭率是多少？"
• "P08181的功能安全状况如何？"

【风险分析查询】
• "AE1部门的高风险项目有哪些？"
• "显示风险评分大于0.8的项目"
• "哪些项目风险最高？"
• "AE2部门的中风险项目"

【数据筛选查询】
• "测试通过率大于80%的项目"
• "A样件阶段的所有项目"
• "查询客户名称包含'JAC'的项目"
• "风险评分大于0.8且客户是JAC的项目"

🔧 支持的操作：
• 筛选：根据部门、客户、风险等级等条件筛选
• 排序：按风险评分、测试通过率等字段排序
• 追问：基于上次查询结果进行进一步分析

📝 使用技巧：
1. 使用自然语言描述您的需求
2. 可以指定项目编号（如P08379）进行精确查询
3. 可以组合多个条件进行复杂查询
4. 支持中文查询，不区分大小写
5. 可以基于上次查询结果进行追问
"""
                    return help_text
            elif "NormalTemplate" in classify:
                result, generated_code, full_result = self.callAgent(get_products_df(), query["input"], normal_prompt)
            elif "DetailInfoTemplate" in classify:
                # 检测是否为文本字段查询（需要两阶段处理）
                text_field_query_result = self.detect_and_handle_text_field_query(query["input"], messages)

                if text_field_query_result is not None:
                    # 文本字段查询，转换为DataFrame格式
                    import pandas as pd
                    import re
                    # 从问题中提取项目编号
                    project_pattern = r'P\d{5}(?:\.\d+)?'
                    project_match = re.search(project_pattern, query["input"])
                    project_no = project_match.group(0) if project_match else ''
                    
                    if project_no:
                        # 获取项目信息以获取Department
                        df = get_products_df()
                        project_info = df[df['ProjectNo'] == project_no]
                        
                        if len(project_info) > 0:
                            department = project_info.iloc[0]['Department']
                            # 创建包含Department和ProjectNo的DataFrame
                            result = pd.DataFrame([{
                                'Department': department,
                                'ProjectNo': project_no,
                                'Answer': text_field_query_result
                            }])
                        else:
                            # 如果找不到项目信息，仍然包含ProjectNo列
                            result = pd.DataFrame([{
                                'Department': '',
                                'ProjectNo': project_no,
                                'Answer': text_field_query_result
                            }])
                    else:
                        # 如果没有提取到项目编号，仍然包含Department和ProjectNo列
                        result = pd.DataFrame([{
                            'Department': '',
                            'ProjectNo': '',
                            'Answer': text_field_query_result
                        }])
                    generated_code = None  # 没有生成代码
                    full_result = result  # DataFrame结果
                else:
                    # 普通详细信息查询（如Project_Summary），使用原流程
                    result, generated_code, full_result = self.callAgent(get_products_df(), query["input"], detail_info_prompt)
                    # 确保结果是DataFrame格式
                    import pandas as pd
                    if result is not None and not isinstance(result, pd.DataFrame):
                        result = pd.DataFrame([{
                            'Result': str(result)
                        }])
            elif "RiskRateTemplate" in classify:
                result, generated_code, full_result = self.callAgent(get_products_df(), query["input"], risk_info_prompt)
            elif "RiskDetermination" in classify:
                return "风险模型\r\n\r\n![risk_theresholds](https://open-webui.cn.kostal.int/imagebed/sample/pic_risk_theresholds_sample.png)"
            elif "FollowUpTemplate" in classify:
                # 首先检测是否为文本字段追问
                text_field_query_result = self.detect_and_handle_text_field_query(query["input"], messages, from_followup=True)

                if text_field_query_result is not None:
                    # 文本字段追问，直接返回LLM理解结果
                    result = text_field_query_result
                    generated_code = None
                    full_result = None
                else:
                    # 常规DataFrame追问
                    if not self.conversation_manager.has_previous_result_full():
                        return "没有可用的上次查询结果，请先进行一次查询。"
                    result, generated_code, full_result = self.callAgent(get_products_df(), query["input"], followup_prompt, use_last_result=True)
            elif "ContinueTemplate" in classify:
                if self.df_chunks:
                    self.current_chunk += 1
                    if self.current_chunk > 0 and self.current_chunk < len(self.df_chunks):
                        return self.df_chunks[self.current_chunk]
                    else:
                        self.current_chunk = 0
                        self.df_chunks = None
                        return "数据已经全部提供，没有更多数据了"
                return "您需要先查询内容"
            else:
                return """👋 你好！我是KOSTAL项目管理智能助手，可通过自然语言查询项目数据，助力项目决策与风险管控。"""
                
            # 保存查询结果用于追问（除了继续查询）
            if classify not in ["ContinueTemplate", "Other_template", "RiskDetermination"]:
                is_followup = (classify == "FollowUpTemplate")  # 判断是否为追问
                self.conversation_manager.save_query_result(
                    result=result,
                    full_result=full_result,
                    is_followup=is_followup
                )
                
                # 保存对话历史到history列表中
                self.conversation_manager.add_message("user", query["input"])
                # 将结果转换为字符串保存
                result_str = str(result) if result is not None else "无结果"
                self.conversation_manager.add_message("assistant", result_str)
                
        except Exception as e:
            print(f"查询执行错误: {e}")
            return f"查询执行出错: {str(e)}"
            
        return result

    def gen_prefix(self):
        if not hasattr(self, 'df_raw_total') or self.df_raw_total is None:
            return ""
        start_index = self.current_chunk * self.chunk_size + 1  # +1 因为用户习惯从1开始计数
        end_index = min((self.current_chunk + 1) * self.chunk_size, self.df_raw_total)
        template_zh = "当前显示: {}/{}:\n"
        template_en = "Currently showing: {}/{}:\n"
        return f"{template_zh.format(end_index, self.df_raw_total)}{template_en.format(end_index, self.df_raw_total)}"
    
    def gen_suffix(self):
        if not hasattr(self, 'df_chunks') or self.df_chunks is None or len(self.df_chunks) <= 1:
            return ""
        if len(self.df_chunks) == 1 or self.current_chunk + 1 == len(self.df_chunks):
            return ""
        template_zh = "\r\n\r\n第{}页/共{}页，输入：\'继续\',显示后续结果。"
        template_en = "Page {}/{}, Input: \'Continue\' to show more results."
        return f"{template_zh.format(self.current_chunk + 1, len(self.df_chunks))}\r\n{template_en.format(self.current_chunk + 1, len(self.df_chunks))}"
    
    async def on_startup(self):
        pass
    
    async def on_shutdown(self):
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str,  Generator, Iterator]:
        # 端到端延迟跟踪
        e2e_start_time = time.time()
        
        # 扩展的自动请求过滤器
        # 只在release环境中记录telemetry
        if os.environ.get("KOSTAL_ENV") == "release":
            if not re.search(r"<chat_history>.*?</chat_history>", messages[-1]['content'], re.DOTALL):
                telemetry = KostalTelemetryClient(tool_name="Project AI", tool_version="3.0.0", user=body['user']['name'])

                telemetry.start_event("Project AI Client")
                telemetry.event("Project_AI_event", event_message="query", event_metadata={"msg": user_message})
                telemetry.stop_event("Project AI client finished")
        auto_request_indicators = [
            "### Task:",
            "follow-up questions",
            "Generate a concise",
            "summarizing the chat history",
            "JSON format:",
            "chat_history>",
            "### Guidelines:",
            "### Output:"
        ]
        
        # 检查是否为自动请求
        if any(indicator in user_message for indicator in auto_request_indicators):
            print("检测到OpenWebUI自动请求，忽略处理")
            return ""
            
        pd.set_option('display.max_columns', None)  # 显示所有列
        pd.set_option('display.max_rows', None)     # 显示所有行
        pd.set_option('display.width', None)         # 自动调整宽度
        pd.set_option('display.max_colwidth', None)  # 显示完整列内容
        
        print(f"user_message##:{user_message}")

        try:
            self.user_logger.log(body['user']['name'], user_message)
        except Exception as e:
            print(f"[警告] 用户提问日志写入失败: {e}")

        user_message += "/no_think"
        response = self.prompt_router({"input": user_message}, messages)
        
        # 记录端到端延迟
        e2e_time = time.time() - e2e_start_time
        self.e2e_times.append(e2e_time)
        self.e2e_count += 1

        print("response的类型：", type(response))
        if isinstance(response, str):
            return self.clean_think_tags_from_answer(response)
        elif isinstance(response, pd.DataFrame):
            # 截断所有长文本字段后再转换为markdown
            response_truncated = truncate_long_text_fields(response, Config.LONG_TEXT_MAX_LENGTH)
            markdown_text = response_truncated.to_markdown(index=False)
            markdown_text = re.sub(r'-{4,}', '---', markdown_text)
            markdown_text = re.sub(r'\|\s+([^|]*?)\s+\|', r'| \1 |', markdown_text)
            markdown_text = re.sub(r'\s{2,}', ' ', markdown_text) 
            if hasattr(self, 'df_chunks') and self.df_chunks is not None:
                return self.gen_prefix() + markdown_text + self.gen_suffix()
            else:
                return markdown_text
        elif isinstance(response, np.ndarray) or isinstance(response, pd.Series) :
            mydf = pd.DataFrame(response)
            # 截断所有长文本字段后再转换为markdown
            mydf_truncated = truncate_long_text_fields(mydf, Config.LONG_TEXT_MAX_LENGTH)
            markdown_text = mydf_truncated.to_markdown(index=False)
            markdown_text = re.sub(r'-{4,}', '---', markdown_text)
            markdown_text = re.sub(r'\|\s+([^|]*?)\s+\|', r'| \1 |', markdown_text)
            markdown_text = re.sub(r'\s{2,}', ' ', markdown_text) 
            return markdown_text
        elif isinstance(response, int) or isinstance(response, float):
            return str(response)
        elif isinstance(response, dict):
            return response
        else:
            print("这个 response的类型我不会处理：",type(response))
            return "我是KOSTAL项目专家，你能问我项目相关的数据"
    
    def get_system_status(self):
        """获取系统状态信息"""
        status_info = {
            "系统名称": self.name,
            "是否有上次查询结果": self.conversation_manager.has_previous_result_full(),
            "当前分页信息": f"第{self.current_chunk + 1}页/共{len(self.df_chunks)}页" if self.df_chunks else "无分页数据",
            "数据源表": "Project_List",
            "数据行数": len(get_products_df())
        }

        return "\n".join([f"{key}: {value}" for key, value in status_info.items()])
    
    def get_metrics(self):
        """获取性能指标"""
        avg_llm_time = sum(self.llm_call_times) / len(self.llm_call_times) if self.llm_call_times else 0.0
        avg_e2e_time = sum(self.e2e_times) / len(self.e2e_times) if self.e2e_times else 0.0
        
        return {
            "token消耗数": {
                "总token数": self.total_tokens,
                "提示token数": self.prompt_tokens,
                "完成token数": self.completion_tokens
            },
            "LLM调用": {
                "调用次数": self.llm_call_count,
                "平均延迟(秒)": round(avg_llm_time, 3),
                "总延迟(秒)": round(sum(self.llm_call_times), 3)
            },
            "端到端延迟": {
                "查询次数": self.e2e_count,
                "平均延迟(秒)": round(avg_e2e_time, 3),
                "总延迟(秒)": round(sum(self.e2e_times), 3)
            },
            "数据库查询": db_reader.get_metrics()
        }
    
    def get_conversation_summary(self):
        """获取对话历史摘要"""
        return ConversationManager.extract_context_from_messages(self.conversation_manager.history)
    
    def clear_conversation_history(self):
        """清空对话历史和重置分页相关变量"""
        self.conversation_manager.clear_history()
        # 重置分页变量
        self.df_chunks = []
        self.current_chunk = 0
        return "对话历史已清空，分页状态已重置"

# 测试代码 - 仅在直接运行此文件时执行
def main():
    """本地测试函数 - 支持多轮对话测试"""
    import asyncio
    
    # 创建 Pipeline 实例，设置历史记录长度为8
    pipeline = Pipeline(max_history_length=8)
    
    # 初始化 pipeline
    async def init_and_test():
        # 初始化
        await pipeline.on_startup()
        
        print("=" * 60)
        print("  KOSTAL Project AI 3.0 - MultiRound Edition 测试")
        print("=" * 60)
        print("系统功能：")
        print("- 多轮对话支持")
        print("- 基于数量的历史记录管理")
        print("- 上下文追问处理")
        print("- 分页数据显示")
        print("- 智能分类器")
        print("- 改进的pandas布尔运算处理")
        print("- 风险评定规则支持")
        print("-" * 60)
        
        # 多轮对话测试用例
        conversation_tests = [
            # 第一轮：基础查询
            " AE1部门的项目有哪些？",
            # 第二轮：基于上次结果的追问
            "这些项目中风险最高的是哪个？",
            # 第三轮：继续基于结果筛选
            "其中HKMC客户的项目有几个？",
            # 第四轮：基于项目的查询
            "这些项目的平均测试通过率是多少？",
        ]
        print("=== 多轮对话测试开始 ===")
        print("模拟用户与AI的连续对话...")
        print("-" * 50)
        # 准备参数
        model_id = "test_model"
        messages = []
        body = {'user':{'name':'li054'}}
        
        # 执行多轮对话测试
        for round_num, query in enumerate(conversation_tests, 1):
            print(f"\n【第{round_num}轮对话】")
            print(f"用户: {query}")
            print("-" * 30)    
            # 执行查询
            result = pipeline.pipe(query, model_id, messages, body)
            # 显示结果（截取前300字符以便查看）
            result_preview = str(result)[:300] + "..." if len(str(result)) > 300 else str(result)
            print(f"AI回复:")
            print(result_preview)
            print("-" * 50)
            
            # 模拟短暂停顿
            import time
            time.sleep(0.3)
        
        # 显示系统状态
        print(f"\n=== 系统状态信息 ===")
        print(pipeline.get_system_status())
        
        # 显示对话历史摘要
        print(f"\n=== 对话历史摘要 ===")
        print(pipeline.get_conversation_summary())
        
        # 测试历史记录管理
        print(f"\n=== 测试历史记录管理 ===")
        print("清空前历史记录数量:", len(pipeline.conversation_manager.history))
        # 分析：历史记录数量应为查询次数，而不是0，表明历史记录未正确保存
        if len(pipeline.conversation_manager.history) == 0:
            print("[警告] 历史记录未正确保存，应包含查询历史")
        clear_result = pipeline.clear_conversation_history()
        print(clear_result)
        print("清空后历史记录数量:", len(pipeline.conversation_manager.history))
        
        # 测试单独的功能
        print(f"\n=== 功能验证测试 ===")
        
        # 测试基础查询
        test_query = "显示AE2部门的项目"
        print(f"测试基础查询: {test_query}")
        result = pipeline.pipe(test_query, model_id, messages, body)
        print(f"结果类型: {type(result)}")
        print(f"结果预览: {str(result)[:100]}...")
        
        # 测试追问（应该提示没有上次结果）
        test_followup = "其中风险最高的项目是哪个？"
        print(f"\n测试追问查询（无上次结果）: {test_followup}")
        result = pipeline.pipe(test_followup, model_id, messages, body)
        print(f"结果: {result}")
        
        # 先查询，再追问
        print(f"\n先查询再追问测试:")
        base_query = "查询所有风险评分大于0.5的项目"
        print(f"基础查询: {base_query}")
        result1 = pipeline.pipe(base_query, model_id, messages, body)
        print(f"基础查询完成")
        
        followup_query = "这些项目按风险评分排序"
        print(f"追问查询: {followup_query}")
        result2 = pipeline.pipe(followup_query, model_id, messages, body)
        print(f"追问查询完成")
        
        # 测试风险评定规则查询
        print(f"\n测试风险评定规则查询:")
        risk_query = "项目的风险是如何评定的？"
        print(f"风险规则查询: {risk_query}")
        result3 = pipeline.pipe(risk_query, model_id, messages, body)
        print(f"风险规则查询结果: {result3}")
        
        # 清理
        await pipeline.on_shutdown()
        
        print("\n" + "=" * 60)
        print("  多轮对话测试完成")
        print("=" * 60)
        print("测试涵盖功能:")
        print("- 基础项目查询")
        print("- 多轮上下文追问")
        print("- 历史记录管理")
        print("- 分类器智能识别")
        print("- 错误处理机制")
        print("- 系统状态监控")
        print("- 改进的pandas布尔运算")
        print("- 风险评定规则展示")
        print("=" * 60)
    
    # 运行异步测试
    asyncio.run(init_and_test())


# 简化版测试函数（用于快速验证）
def quick_test():
    """快速测试函数 - 整合了 test_questions.txt 和 test_cs_status_queries.txt"""
    pipeline = Pipeline(max_history_length=10)
    print(f"历史记录数: {len(pipeline.conversation_manager.history)}")
    # 测试问题集合
    test_categories = {
        "1. 基础查询测试 (test_questions.txt)": [
            "P08760的基本信息",
            "AE1部门有哪些项目？",
            "VOLVO的项目有哪些？",
            "产品型号是SCM的项目",
        ],

        "2. CS_Status 文本字段查询 (test_cs_status_queries.txt)": [
            "P08181的信息安全需求总数是多少？",
            "P08181信息安全的Review占比是多少？",
            "P08181信息安全的Review占比是否超过30%？",
            "P08181有多少条信息安全Review未通过的需求？",
            "P08181信息安全的测试失败的需求有哪些？",
            "P08181信息安全的MiscFunc-Security中Review未通过的有多少条？",
        ],

        "3. Jira_Overview 文本字段查询": [
            "P08181的JIRA问题关闭率是多少？",
            "P08181有多少个高优先级问题？",
        ],

        "4. 需求管理查询": [
            "P08760的系统需求覆盖率怎么样？",
        ],

        "5. 发布状态查询": [
            "P08760的放行成功率是多少？",
        ],

        "6. 多项目查询": [
            "P0818和P0818的基本信息",
            "AE2部门的所有项目列表",
        ],

        "7. 阶段筛选": [
            "C阶段的项目有哪些？",
        ],

        "8. 复杂条件组合": [
            "AE1部门且客户是VOLVO的项目",
        ],

        "9. 追问测试": [
            "P08181的信息安全需求总数是多少？",
            "这个项目的Review占比是多少？",  # 追问
            "这个项目有多少条测试失败？",  # 追问
        ],
    }

    body = {'user': {'name': 'test_user', 'id': 'test_001'}}

    print("=" * 80)
    print("快速综合测试 - 整合 test_questions.txt 和 test_cs_status_queries.txt")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for category, queries in test_categories.items():
        print(f"\n{'=' * 80}")
        print(f"{category}")
        print(f"{'=' * 80}")

        for i, query in enumerate(queries, 1):
            total_tests += 1
            print(f"\n[{total_tests}] {query}")
            print("-" * 80)

            try:
                result = pipeline.pipe(query, "qwen3", [], body)

                # 检查结果
                if result and len(str(result)) > 0:
                    passed_tests += 1
                    # 截断显示结果
                    result_str = str(result)
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "..."
                    print(f"[成功] 结果: {result_str}")
                else:
                    failed_tests += 1
                    print(f"[失败] 结果为空")

            except Exception as e:
                failed_tests += 1
                print(f"[失败] 查询失败: {e}")

        print()

    # 测试总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests} [成功]")
    print(f"失败: {failed_tests} [失败]")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    print(f"历史记录数: {len(pipeline.conversation_manager.history)}")
    print("=" * 80)


# 专门测试open issue查询的函数
def test_open_issue_queries():
    """测试关于项目open issue数量的查询"""
    pipeline = Pipeline(max_history_length=10)
    print(f"历史记录数: {len(pipeline.conversation_manager.history)}")

    # 测试问题 - 边界情况测试相关查询列表
    open_issue_queries = [
        # 24. 更多排序和分析
        # 24.1 多字段排序
        
        "Nissan客户P08181项目总结",
        "哪个项目的当前风险最高？"
        # "哪个项目风险最高"
    ]
    body = {'user': {'name': 'test_user', 'id': 'test_001'}}
    print("\n" + "=" * 80)
    print("专门测试: Open Issue查询")
    print("=" * 80)
    
    for i, query in enumerate(open_issue_queries, 1):
        print(f"\n[{i}] 查询: {query}")
        print("-" * 80)
        try:
            result = pipeline.pipe(query, "qwen3", [], body)
            # 显示完整结果
            print(f"[成功] 结果:")
            print(result)
            print()
            # 显示该查询的性能指标
            print(f"查询{i}性能指标:")
            print("-" * 80)
            print(f"LLM调用延迟: {pipeline.llm_call_times[-1]:.3f}s")
            print(f"Token消耗数: {pipeline.total_tokens} (总token数)")
            print(f"  - 提示token: {pipeline.prompt_tokens}")
            print(f"  - 完成token: {pipeline.completion_tokens}")
            print("-" * 80)
            print()
            
        except Exception as e:
            print(f"[失败] 查询失败: {e}")
            print()
    
    print("=" * 80)
    print("Open Issue查询测试完成")
    print("=" * 80)
    
    # 显示性能指标
    print("\n" + "=" * 80)
    print("性能指标汇总")
    print("=" * 80)
    metrics = pipeline.get_metrics()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("=" * 80)


#测试函数。变更记录：- 2026-02-09  黄鸿和  新增本测试函数
import json
import random
from datetime import datetime

def load_json_to_list(file_path):
    """
    Args:
        读取JSON文件并将数据存入test_queries列表: param file_path: JSON文件路径
    """
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        print(f'成功读取{len(data_list)}条数据')
    except Exception as e:
        print(f'读取文件失败{e}')
    
    return data_list

def load_all_json_from_dir(dir_path):
    all_json_data = []
    # 检查文件夹是否存在
    if not os.path.exists(dir_path):
        print(f"错误：文件夹 '{dir_path}' 不存在！")
        return all_json_data
    
    for filename in os.listdir(dir_path):
        if filename.endswith('.json'):
            file_path = os.path.join(dir_path, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                all_json_data = json.load(f)
                
    return all_json_data




# 专门测试 意图分类器 prompt_router 的测试函数
def test_prompt_router(max_samples: int | None = None) -> None:
    """
    离线批量测试意图分类器 prompt_router。

    - 从 queries.json 中读取测试样本（每条至少包含 input 和 label 字段）
    - 调用 Pipeline.prompt_router 进行分类
    - 在控制台打印对比结果
    - 将汇总结果写入“分类测试结果/classified_results_时间戳.json”

    Args:
        max_samples: 可选，最多测试多少条样本；None 表示全部测试。
    """
    base_dir = os.path.dirname(__file__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    input_file = os.path.join(base_dir, "queries.json")
    output_dir = os.path.join(base_dir, "分类测试结果")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"classified_results_{timestamp}.json")
    supple_info_dir = os.path.join(base_dir, "信息补充")

    pipeline = Pipeline()
    print("=" * 80)
    print("开始批量测试意图识别分类器（prompt_router）")
    print(f"- 测试数据文件: {input_file}")
    print(f"- 信息补充目录: {supple_info_dir}")
    print(f"- 结果输出文件: {output_file}")

    test_queries = load_json_to_list(input_file)
    random.shuffle(test_queries)
    if max_samples is not None:
        test_queries = test_queries[:max_samples]

    supple_info = load_all_json_from_dir(supple_info_dir)

    collected_results: list[dict] = []

    for index, item in enumerate(test_queries, start=1):
        query_text = item.get("input", "")
        true_label = item.get("label", "")
        print("\n" + "-" * 80)

        classification_result = pipeline.prompt_router(
            query=item,
            messages=[],
            suppleInfo=supple_info,
        )

        pred_label = classification_result.get("classify", "")
        scores = classification_result.get("scores", {})

        is_correct = (pred_label == true_label) if true_label else None

        
        print(f"测试用例 #{index}")
        print(f"  问题 (input): {query_text}")
        print(f"  标准标签 (label): {true_label}")
        print(f"  模型分类 (llm_class): {pred_label}")
        if is_correct is not None:
            print(f"  是否正确: {'✅ 正确' if is_correct else '❌ 错误'}")
        print(f"  scores: {json.dumps(scores, ensure_ascii=False)}")

        collected_results.append(
            {
                "index": index,
                "input": query_text,
                "label": true_label,
                "llm_class": pred_label,
                "scores": scores,
                "is_correct": is_correct,
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(collected_results, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 80)
    print(f"测试完成，共处理 {len(collected_results)} 条样本。结果已写入：{output_file}")

        


if __name__ == "__main__":
    #  test_open_issue_queries()
    test_prompt_router()

