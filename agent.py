"""
智能教育Agent系统 - 命令行版本
功能：题目推荐、智能批改、答疑解惑、错题管理
通过命令行交互使用
"""

import os
import json
import sqlite3
import requests
import sys
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Generator, Tuple
import re
import hashlib
import numpy as np
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import argparse
import readline  # 用于命令行历史记录

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============= 配置 =============
class Config:
    """配置类"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-xaS1LZa4mHHn6t5HQINZk8wweS222b301TVc0RmXs0X9YUB5")
    OPENAI_BASE_URL = "https://api.geekai.pro/v1"
    DATABASE_PATH = "data/education.db"
    CHROMA_PATH = "data/chroma"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    LLM_MODEL = "gpt-3.5-turbo"

    # 创建必要的目录
    os.makedirs("data", exist_ok=True)
    os.makedirs(CHROMA_PATH, exist_ok=True)


# ============= 颜色输出 =============
class Colors:
    """命令行颜色"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_color(text, color=""):
    """带颜色的打印"""
    print(f"{color}{text}{Colors.ENDC}")


def print_header(text):
    """打印标题"""
    print_color("\n" + "="*60, Colors.CYAN)
    print_color(text.center(60), Colors.BOLD + Colors.CYAN)
    print_color("="*60 + "\n", Colors.CYAN)


def print_menu(options, title=None):
    """打印菜单"""
    if title:
        print_header(title)

    for i, (key, desc) in enumerate(options.items()):
        print_color(f"  [{key}] {desc}", Colors.BLUE)
    print()


def print_progress(step, total_steps, description):
    """打印进度条"""
    percentage = (step / total_steps) * 100
    bar_length = 40
    filled = int(bar_length * step / total_steps)
    bar = '█' * filled + '░' * (bar_length - filled)

    print_color(f"\n[{bar}] {percentage:.0f}% - {description}", Colors.CYAN)


# ============= 数据模型 =============
@dataclass
class GradingResult:
    """批改结果"""
    score: int
    feedback: str
    correct_answer: str
    explanation: str
    knowledge_points: List[str]
    detailed_analysis: str
    suggestions: List[str]


@dataclass
class LearningSession:
    """学习会话"""
    session_id: str
    user_id: str
    recommended_questions: List[Dict]
    selected_question: Dict
    user_answer: str
    grading_result: Optional[GradingResult]
    qa_history: List[Dict]
    start_time: datetime
    end_time: Optional[datetime]


# ============= 底层LLM调用 =============
class OpenAIClient:
    """OpenAI API客户端"""

    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(self, messages: List[Dict], model: str = None,
                       temperature: float = 0.7, **kwargs) -> Dict:
        """调用Chat Completion API"""
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": model or Config.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        url = f"{self.base_url}/embeddings"

        payload = {
            "model": Config.EMBEDDING_MODEL,
            "input": text
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {e}")
            return np.random.randn(1536).tolist()


# ============= 简单向量数据库 =============
class SimpleVectorDB:
    """简单的向量数据库实现"""

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.vectors = {}
        self.metadata = {}
        self.index = {}
        self.load()

    def _hash_content(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.md5(content.encode()).hexdigest()

    def add_document(self, content: str, metadata: Dict = None) -> str:
        """添加文档"""
        content_hash = self._hash_content(content)

        if content_hash in self.index:
            return self.index[content_hash]

        doc_id = f"doc_{len(self.vectors)}"
        self.vectors[doc_id] = None
        self.metadata[doc_id] = {
            "content": content,
            "metadata": metadata or {},
            "hash": content_hash
        }
        self.index[content_hash] = doc_id

        self.save()
        return doc_id

    def get_embedding(self, client: OpenAIClient, doc_id: str) -> List[float]:
        """获取文档嵌入"""
        if doc_id not in self.vectors:
            return None

        if self.vectors[doc_id] is None:
            content = self.metadata[doc_id]["content"]
            self.vectors[doc_id] = client.get_embedding(content)

        return self.vectors[doc_id]

    def similarity_search(self, client: OpenAIClient, query: str, k: int = 3) -> List[Dict]:
        """相似性搜索"""
        query_embedding = client.get_embedding(query)

        results = []
        for doc_id, doc_embedding in self.vectors.items():
            if doc_embedding is None:
                doc_embedding = self.get_embedding(client, doc_id)

            if doc_embedding:
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                results.append({
                    "id": doc_id,
                    "content": self.metadata[doc_id]["content"],
                    "metadata": self.metadata[doc_id]["metadata"],
                    "similarity": similarity
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-10)

    def save(self):
        """保存到文件"""
        data = {
            "vectors": {k: (v if v is not None else None) for k, v in self.vectors.items()},
            "metadata": self.metadata,
            "index": self.index
        }

        filepath = os.path.join(self.persist_dir, "vectordb.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        """从文件加载"""
        filepath = os.path.join(self.persist_dir, "vectordb.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.vectors = data.get("vectors", {})
                self.metadata = data.get("metadata", {})
                self.index = data.get("index", {})


# ============= 数据库管理 =============
class DatabaseManager:
    """SQLite数据库管理器"""

    def __init__(self, db_path: str = Config.DATABASE_PATH):
        self.db_path = db_path
        self._init_tables()
        self._init_sample_data()

    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """初始化数据表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 用户表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT 'Student',
                    total_questions INTEGER DEFAULT 0,
                    correct_count INTEGER DEFAULT 0,
                    avg_score REAL DEFAULT 0,
                    level TEXT DEFAULT '初级',
                    learning_path TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 答题记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS answer_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    question TEXT,
                    user_answer TEXT,
                    correct_answer TEXT,
                    score INTEGER,
                    feedback TEXT,
                    explanation TEXT,
                    knowledge_points TEXT,
                    agent_steps TEXT,
                    analysis_result TEXT,
                    session_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            # 错题表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mistakes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    question TEXT,
                    user_answer TEXT,
                    correct_answer TEXT,
                    explanation TEXT,
                    review_count INTEGER DEFAULT 0,
                    mastered BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            # 知识点表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    point_name TEXT UNIQUE,
                    subject TEXT,
                    description TEXT,
                    examples TEXT,
                    difficulty TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 学习会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    questions_count INTEGER DEFAULT 0,
                    avg_score REAL DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            conn.commit()

    def _init_sample_data(self):
        """初始化示例数据"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 示例用户
            cursor.execute("""
                INSERT OR IGNORE INTO users (user_id, name, level) 
                VALUES (?, ?, ?)
            """, ("student_1", "小明", "初级"))

            cursor.execute("""
                INSERT OR IGNORE INTO users (user_id, name, level) 
                VALUES (?, ?, ?)
            """, ("student_2", "小红", "中级"))

            # 示例知识点
            sample_points = [
                ("一元一次方程", "数学", "形如ax+b=0的方程", "2x+5=13", "简单"),
                ("勾股定理", "数学", "直角三角形两条直角边的平方和等于斜边的平方", "a²+b²=c²", "简单"),
                ("导数", "数学", "函数在某一点的变化率", "f'(x)=2x", "中等"),
                ("四则运算", "数学", "加、减、乘、除四种运算", "(3+4)×5-20", "简单"),
                ("三角函数", "数学", "正弦、余弦、正切等函数", "sin(30°)=0.5", "中等"),
                ("平面几何", "数学", "平面图形的性质和计算", "三角形内角和180°", "简单"),
                ("立体几何", "数学", "空间图形的性质和计算", "长方体体积=长×宽×高", "中等"),
            ]

            for point in sample_points:
                cursor.execute("""
                    INSERT OR IGNORE INTO knowledge_points (point_name, subject, description, examples, difficulty)
                    VALUES (?, ?, ?, ?, ?)
                """, point)

            conn.commit()

    def get_user_profile(self, user_id: str) -> Dict:
        """获取用户画像"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()

            if row:
                profile = dict(row)
                profile['learning_path'] = json.loads(profile.get('learning_path', '[]'))
                return profile
            else:
                # 创建新用户
                cursor.execute("""
                    INSERT INTO users (user_id, learning_path) 
                    VALUES (?, ?)
                """, (user_id, '[]'))
                conn.commit()

                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                return dict(row) if row else {}

    def save_answer_record(self, user_id: str, data: Dict, agent_steps: List = None, session_id: str = None):
        """保存答题记录"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO answer_records 
                (user_id, question, user_answer, correct_answer, score, feedback, 
                 explanation, knowledge_points, agent_steps, analysis_result, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                data.get('question', ''),
                data.get('user_answer', ''),
                data.get('correct_answer', ''),
                data.get('score', 0),
                data.get('feedback', ''),
                data.get('explanation', ''),
                json.dumps(data.get('knowledge_points', [])),
                json.dumps(agent_steps or []),
                json.dumps(data.get('detailed_analysis', {})),
                session_id
            ))

            # 如果是错题，保存到错题表
            if data.get('score', 0) < 60:
                cursor.execute("""
                    INSERT OR REPLACE INTO mistakes 
                    (user_id, question, user_answer, correct_answer, explanation, review_count)
                    VALUES (?, ?, ?, ?, ?, COALESCE(
                        (SELECT review_count + 1 FROM mistakes 
                         WHERE user_id = ? AND question = ?), 0))
                """, (
                    user_id,
                    data.get('question', ''),
                    data.get('user_answer', ''),
                    data.get('correct_answer', ''),
                    data.get('explanation', ''),
                    user_id,
                    data.get('question', '')
                ))

            # 更新用户统计
            profile = self.get_user_profile(user_id)
            total_questions = profile.get('total_questions', 0) + 1
            correct_count = profile.get('correct_count', 0) + (1 if data.get('score', 0) >= 60 else 0)
            avg_score = ((profile.get('avg_score', 0) * profile.get('total_questions', 0)) + data.get('score', 0)) / total_questions

            # 根据平均分更新等级
            if avg_score >= 85:
                level = '高级'
            elif avg_score >= 70:
                level = '中级'
            else:
                level = '初级'

            cursor.execute("""
                UPDATE users 
                SET total_questions = ?, correct_count = ?, avg_score = ?, level = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (total_questions, correct_count, avg_score, level, user_id))

            conn.commit()

    def save_learning_session(self, session: LearningSession):
        """保存学习会话"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO learning_sessions
                (session_id, user_id, start_time, end_time, questions_count, avg_score, status, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.user_id,
                session.start_time,
                session.end_time,
                1,  # 每个会话一道题
                session.grading_result.score if session.grading_result else 0,
                'completed' if session.end_time else 'active',
                json.dumps({
                    'question': session.selected_question,
                    'qa_history': session.qa_history
                })
            ))

            conn.commit()

    def get_user_mistakes(self, user_id: str, limit: int = 20) -> List[Dict]:
        """获取用户错题"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM mistakes 
                WHERE user_id = ? AND mastered = FALSE
                ORDER BY review_count DESC, created_at DESC
                LIMIT ?
            """, (user_id, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_knowledge_points(self, subject: str = None) -> List[Dict]:
        """获取知识点"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if subject:
                cursor.execute("SELECT * FROM knowledge_points WHERE subject = ?", (subject,))
            else:
                cursor.execute("SELECT * FROM knowledge_points")

            return [dict(row) for row in cursor.fetchall()]

    def get_answer_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取答题历史"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM answer_records 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (user_id, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_recent_scores(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取最近成绩"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT score, created_at FROM answer_records 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (user_id, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_all_users(self) -> List[Dict]:
        """获取所有用户"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, name, level, total_questions, avg_score FROM users")
            return [dict(row) for row in cursor.fetchall()]


# ============= 智能批改器 =============
class SmartGrader:
    """智能批改器"""

    def __init__(self, llm_client: OpenAIClient, vector_db: SimpleVectorDB):
        self.llm = llm_client
        self.vector_db = vector_db

    def grade_answer(self, question: str, user_answer: str, context: str = "") -> GradingResult:
        """批改答案"""
        # 1. 检索相关知识
        knowledge_context = self._retrieve_knowledge(question)

        # 2. 构建批改提示
        prompt = f"""你是一名专业的老师，请批改学生的答案。

题目：{question}
学生答案：{user_answer}
相关知识点：{knowledge_context}
批改要求：
1. 给出0-100的整数分数
2. 指出答案正确和错误的部分
3. 提供详细的解析
4. 给出学习建议
5. 输出正确的答案
6. 识别涉及的知识点

请以JSON格式返回：
{{
    "score": 分数,
    "feedback": "反馈和建议",
    "correct_answer": "正确答案",
    "explanation": "详细解析",
    "knowledge_points": ["知识点1", "知识点2"],
    "detailed_analysis": "详细的错误分析"
}}"""

        messages = [
            {"role": "system", "content": "你是一名专业的数学老师，擅长批改作业和解释题目。"},
            {"role": "user", "content": prompt}
        ]

        # 3. 调用LLM进行批改
        response = self.llm.chat_completion(messages, temperature=0.1)
        result_text = response["choices"][0]["message"]["content"]

        # 4. 解析结果
        try:
            result_json = self._extract_json(result_text)
        except:
            result_json = self._parse_grading_result(result_text)

        # 5. 创建GradingResult对象
        return GradingResult(
            score=result_json.get("score", 0),
            feedback=result_json.get("feedback", "批改失败"),
            correct_answer=result_json.get("correct_answer", "未知"),
            explanation=result_json.get("explanation", "无解析"),
            knowledge_points=result_json.get("knowledge_points", []),
            detailed_analysis=result_json.get("detailed_analysis", "无详细分析"),
            suggestions=self._generate_suggestions(result_json.get("score", 0))
        )

    def _retrieve_knowledge(self, query: str) -> str:
        """检索相关知识"""
        results = self.vector_db.similarity_search(self.llm, query, k=3)
        if results:
            return "\n".join([f"{i+1}. {r['content']}" for i, r in enumerate(results)])
        return "无相关知识"

    def _extract_json(self, text: str) -> Dict:
        """从文本中提取JSON"""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError("未找到有效的JSON")

    def _parse_grading_result(self, text: str) -> Dict:
        """解析批改结果文本"""
        result = {
            "score": 70,
            "feedback": "请检查计算过程",
            "correct_answer": "需要老师确认",
            "explanation": text,
            "knowledge_points": [],
            "detailed_analysis": text
        }

        score_match = re.search(r'(\d{1,3})\s*分', text)
        if score_match:
            result["score"] = int(score_match.group(1))

        return result

    def _generate_suggestions(self, score: int) -> List[str]:
        """生成学习建议"""
        if score >= 90:
            return ["继续保持，你已经掌握得很好！", "可以尝试更难的题目挑战自己"]
        elif score >= 70:
            return ["注意细节处理", "多加练习同类型题目"]
        else:
            return ["需要重点复习相关知识点", "多做基础练习", "注意审题和计算过程"]


# ============= 题目推荐器 =============
class QuestionRecommender:
    """题目推荐器"""

    def __init__(self, llm_client: OpenAIClient, vector_db: SimpleVectorDB, db_manager: DatabaseManager):
        self.llm = llm_client
        self.vector_db = vector_db
        self.db_manager = db_manager

    def recommend_questions(self, user_id: str, count: int = 5) -> List[Dict]:
        """推荐题目"""
        # 获取用户信息
        profile = self.db_manager.get_user_profile(user_id)
        level = profile.get('level', '初级')

        # 获取用户错题
        mistakes = self.db_manager.get_user_mistakes(user_id, limit=5)

        # 根据用户水平和错题生成题目
        questions = self._generate_questions(level, mistakes, count)

        return questions

    def _generate_questions(self, level: str, mistakes: List[Dict], count: int) -> List[Dict]:
        """生成题目"""
        difficulty_map = {
            '初级': '简单',
            '中级': '中等',
            '高级': '困难'
        }

        difficulty = difficulty_map.get(level, '简单')

        # 预定义的题库
        question_bank = {
            '简单': [
                {'question': '求解方程：2x + 5 = 13', 'type': '一元一次方程', 'difficulty': '简单'},
                {'question': '计算：(3+4) × 5 - 20', 'type': '四则运算', 'difficulty': '简单'},
                {'question': '计算圆的面积，半径r=5', 'type': '平面几何', 'difficulty': '简单'},
                {'question': '计算：√16 + 3²', 'type': '四则运算', 'difficulty': '简单'},
                {'question': '求解方程：3x - 7 = 8', 'type': '一元一次方程', 'difficulty': '简单'},
            ],
            '中等': [
                {'question': '已知三角形ABC，AB=3, AC=4, BC=5，判断三角形类型', 'type': '平面几何', 'difficulty': '中等'},
                {'question': '求导数：f(x) = x² + 3x - 5', 'type': '导数', 'difficulty': '中等'},
                {'question': '解不等式：2x - 7 > 3', 'type': '不等式', 'difficulty': '中等'},
                {'question': '分解因式：x² - 4', 'type': '因式分解', 'difficulty': '中等'},
                {'question': '解方程组：{x + y = 5, x - y = 1}', 'type': '方程组', 'difficulty': '中等'},
            ],
            '困难': [
                {'question': '求二次函数y=x²+2x+1的顶点坐标', 'type': '二次函数', 'difficulty': '困难'},
                {'question': '证明：对于任意正整数n，n³-n能被6整除', 'type': '数论', 'difficulty': '困难'},
                {'question': '求函数f(x)=x³-3x²+2的极值点', 'type': '导数应用', 'difficulty': '困难'},
                {'question': '解三角方程：sin(2x) = cos(x)', 'type': '三角函数', 'difficulty': '困难'},
                {'question': '计算定积分：∫(0到π) sin²(x)dx', 'type': '积分', 'difficulty': '困难'},
            ]
        }

        # 根据难度选择题目
        available_questions = question_bank.get(difficulty, question_bank['简单'])

        # 如果有错题，优先推荐相关题型
        if mistakes:
            # 这里简化处理，实际应该更智能地匹配题型
            return available_questions[:count]

        return available_questions[:count]


# ============= 完整学习流程管理 =============
class LearningFlowManager:
    """完整学习流程管理器"""

    def __init__(self, llm_client: OpenAIClient, vector_db: SimpleVectorDB,
                 db_manager: DatabaseManager, grader: SmartGrader):
        self.llm = llm_client
        self.vector_db = vector_db
        self.db_manager = db_manager
        self.grader = grader
        self.recommender = QuestionRecommender(llm_client, vector_db, db_manager)
        self.current_session = None

    def start_learning_flow(self, user_id: str):
        """开始完整的学习流程"""
        # 创建新的学习会话
        session_id = f"session_{user_id}_{int(time.time())}"
        self.current_session = LearningSession(
            session_id=session_id,
            user_id=user_id,
            recommended_questions=[],
            selected_question={},
            user_answer="",
            grading_result=None,
            qa_history=[],
            start_time=datetime.now(),
            end_time=None
        )

        # 清屏并显示欢迎信息
        self.clear_screen()
        print_header("🎯 智能学习流程")
        print_color("欢迎进入完整学习流程！我们将按以下步骤进行：", Colors.CYAN)
        print_color("1️⃣  推荐适合的题目", Colors.YELLOW)
        print_color("2️⃣  选择并练习题目", Colors.YELLOW)
        print_color("3️⃣  智能批改答案", Colors.YELLOW)
        print_color("4️⃣  答疑解惑", Colors.YELLOW)
        print()
        input(f"{Colors.GREEN}按回车键开始...{Colors.ENDC}")

        # 执行流程
        try:
            # 步骤1：推荐题目
            print_progress(1, 4, "推荐题目")
            questions = self._step1_recommend_questions(user_id)

            if not questions:
                print_color("未能获取题目，请稍后再试。", Colors.RED)
                return

            # 步骤2：选择并练习题目
            print_progress(2, 4, "练习题目")
            selected_question = self._step2_select_and_practice(questions)

            if not selected_question:
                return

            # 步骤3：批改答案
            print_progress(3, 4, "批改答案")
            grading_result = self._step3_grade_answer()

            # 步骤4：答疑解惑
            print_progress(4, 4, "答疑解惑")
            self._step4_qa_session(grading_result)

            # 保存会话
            self.current_session.end_time = datetime.now()
            self.db_manager.save_learning_session(self.current_session)

            # 显示学习总结
            self._show_learning_summary()

        except KeyboardInterrupt:
            print_color("\n\n学习流程已中断。", Colors.YELLOW)
        except Exception as e:
            print_color(f"\n发生错误：{str(e)}", Colors.RED)

    def _step1_recommend_questions(self, user_id: str) -> List[Dict]:
        """步骤1：推荐题目"""
        self.clear_screen()
        print_header("📚 步骤1：题目推荐")

        # 获取用户信息
        profile = self.db_manager.get_user_profile(user_id)
        print_color(f"👤 用户：{profile.get('name', 'N/A')}", Colors.CYAN)
        print_color(f"📊 等级：{profile.get('level', '初级')}", Colors.CYAN)
        print_color(f"💯 平均分：{profile.get('avg_score', 0):.1f}", Colors.CYAN)
        print()

        print_color("🤖 AI正在为你推荐合适的题目...", Colors.YELLOW)

        # 推荐题目
        questions = self.recommender.recommend_questions(user_id, count=5)
        self.current_session.recommended_questions = questions

        print_color(f"\n✨ 为你推荐了 {len(questions)} 道题目：", Colors.GREEN)
        for i, q in enumerate(questions, 1):
            difficulty_color = {
                '简单': Colors.GREEN,
                '中等': Colors.YELLOW,
                '困难': Colors.RED
            }.get(q.get('difficulty', ''), Colors.YELLOW)

            print_color(f"\n  [{i}] {q['question']}", Colors.CYAN)
            print_color(f"      类型：{q['type']} | 难度：{difficulty_color}{q['difficulty']}{Colors.ENDC}", Colors.YELLOW)

        print()
        return questions

    def _step2_select_and_practice(self, questions: List[Dict]) -> Optional[Dict]:
        """步骤2：选择并练习题目"""
        print()
        print_color("请选择要练习的题目编号 (1-5)，或输入 'q' 退出：", Colors.CYAN)

        while True:
            choice = input(f"{Colors.GREEN}> {Colors.ENDC}")

            if choice.lower() == 'q':
                return None

            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(questions):
                    selected = questions[choice_num - 1]
                    self.current_session.selected_question = selected

                    self.clear_screen()
                    print_header("✏️ 步骤2：练习题目")
                    print_color(f"题目：{selected['question']}", Colors.CYAN + Colors.BOLD)
                    print_color(f"类型：{selected['type']} | 难度：{selected['difficulty']}", Colors.YELLOW)
                    print()

                    # 获取用户答案
                    print_color("请输入你的答案（可以输入多行，以单独一行的'END'结束）：", Colors.CYAN)
                    answer_lines = []
                    while True:
                        line = input(f"{Colors.YELLOW}> {Colors.ENDC}")
                        if line.upper() == 'END':
                            break
                        answer_lines.append(line)

                    self.current_session.user_answer = "\n".join(answer_lines)

                    if not self.current_session.user_answer.strip():
                        print_color("答案不能为空，请重新输入。", Colors.RED)
                        continue

                    return selected
                else:
                    print_color("无效的选择，请输入1-5之间的数字。", Colors.RED)
            except ValueError:
                print_color("无效的输入，请输入数字。", Colors.RED)

    def _step3_grade_answer(self) -> GradingResult:
        """步骤3：批改答案"""
        self.clear_screen()
        print_header("📊 步骤3：智能批改")

        print_color("🤖 AI正在批改你的答案，请稍候...", Colors.YELLOW)

        # 批改答案
        result = self.grader.grade_answer(
            self.current_session.selected_question['question'],
            self.current_session.user_answer
        )

        self.current_session.grading_result = result

        # 显示批改结果
        print()
        self._display_grading_result(result)

        # 保存到数据库
        data = {
            "question": self.current_session.selected_question['question'],
            "user_answer": self.current_session.user_answer,
            "correct_answer": result.correct_answer,
            "score": result.score,
            "feedback": result.feedback,
            "explanation": result.explanation,
            "knowledge_points": result.knowledge_points,
            "detailed_analysis": result.detailed_analysis
        }

        self.db_manager.save_answer_record(
            self.current_session.user_id,
            data,
            session_id=self.current_session.session_id
        )

        input(f"\n{Colors.GREEN}按回车键继续到答疑环节...{Colors.ENDC}")
        return result

    def _step4_qa_session(self, grading_result: GradingResult):
        """步骤4：答疑解惑"""
        self.clear_screen()
        print_header("💬 步骤4：答疑解惑")

        # 根据批改结果生成初始问题建议
        if grading_result.score < 60:
            print_color("看起来这道题有些困难，让我来帮助你理解！", Colors.YELLOW)
            suggestions = [
                "解释一下这道题的解题思路",
                "详细说明计算步骤",
                "介绍相关的知识点",
                "给我类似的练习题"
            ]
        elif grading_result.score < 90:
            print_color("做得不错！还有一些细节可以改进。", Colors.YELLOW)
            suggestions = [
                "指出容易出错的地方",
                "提供更优的解法",
                "解释知识点的深层含义"
            ]
        else:
            print_color("🎉 太棒了！你已经很好地掌握了这道题。", Colors.GREEN)
            suggestions = [
                "提供更难的挑战题",
                "探讨知识点的扩展应用",
                "分享学习技巧"
            ]

        print_color("\n你可能想问的问题：", Colors.CYAN)
        for i, s in enumerate(suggestions, 1):
            print_color(f"  [{i}] {s}", Colors.YELLOW)

        print()
        print_color("输入你的问题（输入'q'结束答疑）：", Colors.CYAN)

        # 初始化对话
        messages = [
            {"role": "system", "content": f"""你是一位专业的AI教师助手。
学生刚完成了以下题目：
题目：{self.current_session.selected_question['question']}
学生答案：{self.current_session.user_answer}
正确答案：{grading_result.correct_answer}
得分：{grading_result.score}
批改反馈：{grading_result.feedback}

请根据学生的表现，耐心、详细地回答学生的问题，帮助学生理解和掌握相关知识。"""}
        ]

        while True:
            user_input = input(f"{Colors.GREEN}你: {Colors.ENDC}")

            if user_input.lower() == 'q':
                break

            # 检查是否选择建议问题
            try:
                choice_num = int(user_input)
                if 1 <= choice_num <= len(suggestions):
                    user_input = suggestions[choice_num - 1]
                    print_color(f"你: {user_input}", Colors.GREEN)
            except ValueError:
                pass

            messages.append({"role": "user", "content": user_input})

            print(f"{Colors.YELLOW}AI正在思考...{Colors.ENDC}")

            # 调用AI
            response = self.llm.chat_completion(messages, temperature=0.7)
            ai_response = response["choices"][0]["message"]["content"]

            messages.append({"role": "assistant", "content": ai_response})

            print(f"{Colors.CYAN}AI: {Colors.ENDC}{ai_response}\n")

            # 保存到会话
            self.current_session.qa_history.append({
                "question": user_input,
                "answer": ai_response
            })

    def _display_grading_result(self, result: GradingResult):
        """显示批改结果"""
        # 显示分数
        if result.score >= 90:
            score_color = Colors.GREEN
            score_emoji = "🎉"
        elif result.score >= 70:
            score_color = Colors.YELLOW
            score_emoji = "👍"
        else:
            score_color = Colors.RED
            score_emoji = "💪"

        print_color(f"{score_emoji} 得分: {result.score}/100 {score_emoji}", score_color + Colors.BOLD)
        print()

        # 显示反馈
        print_color("📋 反馈：", Colors.CYAN)
        print_color(f"  {result.feedback}", Colors.YELLOW)
        print()

        # 显示正确答案
        print_color("✅ 正确答案：", Colors.CYAN)
        print_color(f"  {result.correct_answer}", Colors.GREEN)
        print()

        # 显示解析
        print_color("🔍 详细解析：", Colors.CYAN)
        for line in result.explanation.split('\n'):
            print_color(f"  {line}", Colors.YELLOW)
        print()

        # 显示知识点
        if result.knowledge_points:
            print_color("📚 涉及知识点：", Colors.CYAN)
            for i, point in enumerate(result.knowledge_points, 1):
                print_color(f"  {i}. {point}", Colors.YELLOW)

    def _show_learning_summary(self):
        """显示学习总结"""
        self.clear_screen()
        print_header("📈 学习总结")

        session = self.current_session

        # 计算学习时长
        duration = (session.end_time - session.start_time).total_seconds() / 60

        print_color(f"⏱️  学习时长：{duration:.1f} 分钟", Colors.CYAN)
        print_color(f"📝 练习题目：{session.selected_question['question'][:50]}...", Colors.CYAN)
        print_color(f"💯 得分：{session.grading_result.score}/100", Colors.CYAN)
        print()

        # 知识点掌握情况
        if session.grading_result.knowledge_points:
            print_color("📚 涉及知识点：", Colors.CYAN)
            for point in session.grading_result.knowledge_points:
                print_color(f"  • {point}", Colors.YELLOW)
            print()

        # 学习建议
        print_color("💡 学习建议：", Colors.CYAN)
        for suggestion in session.grading_result.suggestions:
            print_color(f"  • {suggestion}", Colors.YELLOW)

        # 答疑记录
        if session.qa_history:
            print()
            print_color(f"💬 答疑记录：共 {len(session.qa_history)} 个问题", Colors.CYAN)

        print()
        print_color("🎯 继续努力，你会越来越棒的！", Colors.GREEN + Colors.BOLD)

        input(f"\n{Colors.YELLOW}按回车键返回主菜单...{Colors.ENDC}")

    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')


# ============= 命令行界面 =============
class CommandLineInterface:
    """命令行界面"""

    def __init__(self):
        # 初始化组件
        self.llm_client = OpenAIClient(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL
        )
        self.vector_db = SimpleVectorDB(Config.CHROMA_PATH)
        self.db_manager = DatabaseManager()
        self.grader = SmartGrader(self.llm_client, self.vector_db)
        self.flow_manager = LearningFlowManager(
            self.llm_client,
            self.vector_db,
            self.db_manager,
            self.grader
        )

        # 当前用户
        self.current_user = "student_1"

        # 加载初始数据
        self._load_initial_data()

        # 对话历史
        self.conversation_history = []

    def _load_initial_data(self):
        """加载初始数据到向量数据库"""
        # 加载知识点
        knowledge_points = self.db_manager.get_knowledge_points()
        for point in knowledge_points:
            content = f"{point['point_name']}: {point['description']}\n示例: {point['examples']}"
            self.vector_db.add_document(content, {
                "type": "knowledge_point",
                "subject": point.get("subject"),
                "difficulty": point.get("difficulty")
            })

        # 加载示例题目
        sample_questions = [
            "求解方程：2x + 5 = 13",
            "计算：(3+4) × 5 - 20",
            "已知三角形ABC，AB=3, AC=4, BC=5，判断三角形类型",
            "求导数：f(x) = x² + 3x - 5",
            "解不等式：2x - 7 > 3",
            "计算圆的面积，半径r=5",
            "计算：√16 + 3²",
            "分解因式：x² - 4",
            "解方程组：{x + y = 5, x - y = 1}",
            "求二次函数y=x²+2x+1的顶点坐标"
        ]

        for i, question in enumerate(sample_questions):
            self.vector_db.add_document(question, {
                "type": "sample_question",
                "subject": "数学",
                "index": i
            })

    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_welcome(self):
        """打印欢迎信息"""
        self.clear_screen()
        print_color("="*60, Colors.CYAN)
        print_color("🤖 智能教育Agent系统 - 命令行版本", Colors.BOLD + Colors.CYAN)
        print_color("="*60, Colors.CYAN)
        print_color("📚 功能：完整学习流程、智能批改、答疑解惑、错题管理", Colors.GREEN)
        print_color(f"👤 当前用户：{self.current_user}", Colors.YELLOW)
        print_color(f"📁 数据路径：{os.path.abspath('data/')}", Colors.YELLOW)
        print_color("="*60, Colors.CYAN)
        print()

    def show_main_menu(self):
        """显示主菜单"""
        while True:
            self.print_welcome()

            menu_options = {
                "1": "🎯 开始完整学习流程（推荐）",
                "2": "📝 单独批改题目",
                "3": "💬 AI聊天答疑",
                "4": "📚 查看推荐题目",
                "5": "❌ 查看我的错题",
                "6": "📈 查看学习统计",
                "7": "📜 查看答题历史",
                "8": "🔄 切换用户",
                "9": "⚙️  系统信息",
                "0": "👋 退出系统"
            }

            print_menu(menu_options, "主菜单")

            choice = input(f"{Colors.GREEN}请选择操作 (0-9): {Colors.ENDC}")

            if choice == "1":
                self.flow_manager.start_learning_flow(self.current_user)
            elif choice == "2":
                self.grade_answer_interactive()
            elif choice == "3":
                self.chat_assistant()
            elif choice == "4":
                self.recommend_questions()
            elif choice == "5":
                self.view_mistakes()
            elif choice == "6":
                self.view_statistics()
            elif choice == "7":
                self.view_history()
            elif choice == "8":
                self.switch_user()
            elif choice == "9":
                self.system_info()
            elif choice == "0":
                print_color("感谢使用，再见！", Colors.GREEN)
                sys.exit(0)
            else:
                print_color("无效选择，请重新输入！", Colors.RED)
                time.sleep(1)

    def grade_answer_interactive(self):
        """交互式批改答案"""
        self.clear_screen()
        print_header("📝 智能批改")

        # 获取题目
        print_color("请输入题目（输入'q'返回主菜单）：", Colors.CYAN)
        question = input(f"{Colors.YELLOW}> {Colors.ENDC}")

        if question.lower() == 'q':
            return

        if not question.strip():
            print_color("题目不能为空！", Colors.RED)
            time.sleep(1)
            return self.grade_answer_interactive()

        # 获取答案
        print_color("\n请输入你的答案（输入'q'返回主菜单）：", Colors.CYAN)
        print_color("（可以输入多行，以单独一行的'END'结束）", Colors.YELLOW)

        answer_lines = []
        while True:
            line = input(f"{Colors.YELLOW}> {Colors.ENDC}")
            if line.upper() == 'END':
                break
            elif line.upper() == 'Q':
                return
            answer_lines.append(line)

        user_answer = "\n".join(answer_lines)

        if not user_answer.strip():
            print_color("答案不能为空！", Colors.RED)
            time.sleep(1)
            return self.grade_answer_interactive()

        # 显示处理中
        print_color("\n🔍 AI正在批改你的答案，请稍候...", Colors.YELLOW)

        try:
            # 调用批改器
            result = self.grader.grade_answer(question, user_answer)

            # 显示结果
            self.clear_screen()
            print_header("📊 批改结果")

            # 显示分数
            if result.score >= 90:
                score_color = Colors.GREEN
                score_emoji = "🎉"
            elif result.score >= 70:
                score_color = Colors.YELLOW
                score_emoji = "👍"
            else:
                score_color = Colors.RED
                score_emoji = "💪"

            print_color(f"{score_emoji} 得分: {result.score}/100 {score_emoji}", score_color + Colors.BOLD)
            print()

            # 显示反馈
            print_color("📋 反馈：", Colors.CYAN)
            print_color(f"  {result.feedback}", Colors.YELLOW)
            print()

            # 显示正确答案
            print_color("✅ 正确答案：", Colors.CYAN)
            print_color(f"  {result.correct_answer}", Colors.GREEN)
            print()

            # 显示解析
            print_color("🔍 详细解析：", Colors.CYAN)
            for line in result.explanation.split('\n'):
                print_color(f"  {line}", Colors.YELLOW)
            print()

            # 显示知识点
            if result.knowledge_points:
                print_color("📚 涉及知识点：", Colors.CYAN)
                for i, point in enumerate(result.knowledge_points, 1):
                    print_color(f"  {i}. {point}", Colors.YELLOW)
                print()

            # 显示建议
            if result.suggestions:
                print_color("💡 学习建议：", Colors.CYAN)
                for i, suggestion in enumerate(result.suggestions, 1):
                    print_color(f"  {i}. {suggestion}", Colors.YELLOW)
                print()

            # 保存到数据库
            data = {
                "question": question,
                "user_answer": user_answer,
                "correct_answer": result.correct_answer,
                "score": result.score,
                "feedback": result.feedback,
                "explanation": result.explanation,
                "knowledge_points": result.knowledge_points,
                "detailed_analysis": result.detailed_analysis
            }

            self.db_manager.save_answer_record(self.current_user, data)

            print_color(f"📁 记录已保存到数据库，用户：{self.current_user}", Colors.GREEN)

        except Exception as e:
            print_color(f"批改失败：{str(e)}", Colors.RED)

        input(f"\n{Colors.YELLOW}按回车键返回主菜单...{Colors.ENDC}")

    def chat_assistant(self):
        """聊天助手"""
        self.clear_screen()
        print_header("💬 AI学习助手")
        print_color("输入你的问题，AI会为你解答（输入'q'退出）", Colors.CYAN)
        print_color("输入'/history'查看对话历史，'/clear'清空历史", Colors.YELLOW)
        print()

        # 初始化对话历史
        if not hasattr(self, 'chat_history'):
            self.chat_history = [
                {"role": "system", "content": "你是一位AI教师助手，擅长解答学习问题、批改作业、提供学习建议。回答要专业、准确、有帮助。"}
            ]

        while True:
            try:
                user_input = input(f"{Colors.GREEN}你: {Colors.ENDC}")

                if user_input.lower() == 'q':
                    break
                elif user_input == '/history':
                    self._show_chat_history()
                    continue
                elif user_input == '/clear':
                    self.chat_history = [
                        {"role": "system", "content": "你是一位AI教师助手，擅长解答学习问题、批改作业、提供学习建议。回答要专业、准确、有帮助。"}
                    ]
                    print_color("对话历史已清空", Colors.GREEN)
                    continue

                # 添加到历史
                self.chat_history.append({"role": "user", "content": user_input})

                # 显示思考中
                print(f"{Colors.YELLOW}AI正在思考...{Colors.ENDC}")

                # 调用API
                response = self.llm_client.chat_completion(
                    messages=self.chat_history,
                    temperature=0.7
                )

                ai_response = response["choices"][0]["message"]["content"]

                # 添加AI回复到历史
                self.chat_history.append({"role": "assistant", "content": ai_response})

                # 显示AI回复
                print(f"{Colors.CYAN}AI: {Colors.ENDC}{ai_response}\n")

            except KeyboardInterrupt:
                print_color("\n\n返回主菜单...", Colors.YELLOW)
                break
            except Exception as e:
                print_color(f"错误：{str(e)}", Colors.RED)

    def _show_chat_history(self):
        """显示对话历史"""
        print_header("📜 对话历史")
        for msg in self.chat_history[1:]:  # 跳过系统提示
            role = "你" if msg["role"] == "user" else "AI"
            color = Colors.GREEN if msg["role"] == "user" else Colors.CYAN
            print_color(f"{role}: {msg['content'][:100]}...", color)
        print()

    def recommend_questions(self):
        """推荐题目"""
        self.clear_screen()
        print_header("🎯 题目推荐")

        # 获取用户信息
        profile = self.db_manager.get_user_profile(self.current_user)
        level = profile.get('level', '初级')

        recommender = QuestionRecommender(self.llm_client, self.vector_db, self.db_manager)
        questions = recommender.recommend_questions(self.current_user, count=5)

        print_color(f"根据你的水平（{level}），推荐以下题目：", Colors.CYAN)
        print()

        for i, q in enumerate(questions, 1):
            difficulty_color = {
                '简单': Colors.GREEN,
                '中等': Colors.YELLOW,
                '困难': Colors.RED
            }.get(q.get('difficulty', ''), Colors.YELLOW)

            print_color(f"[{i}] {q['question']}", Colors.CYAN)
            print_color(f"    类型：{q['type']} | 难度：{difficulty_color}{q['difficulty']}{Colors.ENDC}", Colors.YELLOW)
            print()

        print_color("💡 提示：使用【开始完整学习流程】功能可以直接练习这些题目", Colors.YELLOW)

        input(f"\n{Colors.YELLOW}按回车键返回主菜单...{Colors.ENDC}")

        def view_mistakes(self):
            """查看错题"""
            self.clear_screen()
            print_header("❌ 我的错题本")

            mistakes = self.db_manager.get_user_mistakes(self.current_user, limit=20)

            if not mistakes:
                print_color("🎉 恭喜！目前没有错题记录。", Colors.GREEN)
            else:
                print_color(f"📊 找到 {len(mistakes)} 道错题：", Colors.CYAN)
                print()

                for i, mistake in enumerate(mistakes, 1):
                    print_color(f"#{i} 错题", Colors.YELLOW)
                    print_color(f"   题目：{mistake.get('question', '')}", Colors.YELLOW)
                    print_color(f"   你的答案：{mistake.get('user_answer', '')[:50]}...", Colors.RED)
                    print_color(f"   正确答案：{mistake.get('correct_answer', '')[:50]}...", Colors.GREEN)
                    print_color(f"   复习次数：{mistake.get('review_count', 0)}", Colors.YELLOW)
                    print_color(f"   时间：{mistake.get('created_at', '')}", Colors.YELLOW)
                    print()

            # 操作菜单
            if mistakes:
                print_color("操作选项：", Colors.CYAN)
                print_color("  [r] 重新练习错题", Colors.GREEN)
                print_color("  [c] 开始完整学习流程（从错题开始）", Colors.BLUE)
                print_color("  [m] 标记为已掌握", Colors.YELLOW)
                print_color("  [回车] 返回主菜单", Colors.YELLOW)

                choice = input(f"\n{Colors.GREEN}请选择操作: {Colors.ENDC}").lower()

                if choice == 'r':
                    self._review_mistake(mistakes)
                elif choice == 'c':
                    # 启动完整学习流程，优先推荐错题相关题目
                    self.flow_manager.start_learning_flow(self.current_user)
                elif choice == 'm':
                    self._mark_mistake_mastered(mistakes)
            else:
                input(f"\n{Colors.YELLOW}按回车键返回主菜单...{Colors.ENDC}")

        def _review_mistake(self, mistakes):
            """重新练习错题"""
            if not mistakes:
                return

            self.clear_screen()
            print_header("📝 重新练习错题")

            print_color("请选择要重新练习的错题编号，或输入0返回：", Colors.CYAN)

            # 显示前5道错题
            display_count = min(5, len(mistakes))
            for i, mistake in enumerate(mistakes[:display_count], 1):
                question = mistake.get('question', '')
                if len(question) > 50:
                    question = question[:47] + "..."
                print_color(f"  [{i}] {question}", Colors.YELLOW)

            choice = input(f"\n{Colors.GREEN}选择: {Colors.ENDC}")

            try:
                choice_num = int(choice)
                if choice_num == 0:
                    return
                elif 1 <= choice_num <= display_count:
                    mistake = mistakes[choice_num - 1]
                    self._practice_mistake(mistake)
            except ValueError:
                print_color("无效的输入", Colors.RED)
                time.sleep(1)

        def _practice_mistake(self, mistake):
            """练习具体错题"""
            self.clear_screen()
            print_header("📝 错题重练")

            question = mistake.get('question', '')
            correct_answer = mistake.get('correct_answer', '')
            previous_answer = mistake.get('user_answer', '')

            print_color(f"题目：{question}", Colors.CYAN + Colors.BOLD)
            print()
            print_color("上次的错误答案：", Colors.RED)
            print_color(f"  {previous_answer[:100]}...", Colors.YELLOW)
            print()
            print_color("参考正确答案：", Colors.GREEN)
            print_color(f"  {correct_answer}", Colors.YELLOW)
            print()

            print_color("请输入你的新答案（以单独一行的'END'结束）：", Colors.CYAN)
            answer_lines = []
            while True:
                line = input(f"{Colors.YELLOW}> {Colors.ENDC}")
                if line.upper() == 'END':
                    break
                answer_lines.append(line)

            new_answer = "\n".join(answer_lines)

            if new_answer.strip():
                print_color("\n🔍 正在批改你的新答案...", Colors.YELLOW)

                try:
                    result = self.grader.grade_answer(question, new_answer)

                    # 显示新成绩
                    print()
                    if result.score >= 90:
                        print_color(f"🎉 太棒了！新得分：{result.score}/100", Colors.GREEN + Colors.BOLD)
                        print_color("你已经掌握了这道题！", Colors.GREEN)
                    elif result.score >= 60:
                        print_color(f"👍 不错！新得分：{result.score}/100", Colors.YELLOW)
                        print_color("继续努力，你正在进步！", Colors.YELLOW)
                    else:
                        print_color(f"💪 新得分：{result.score}/100", Colors.RED)
                        print_color("还需要继续努力哦！", Colors.RED)

                    print()
                    print_color("反馈：", Colors.CYAN)
                    print_color(f"  {result.feedback}", Colors.YELLOW)

                    # 保存新记录
                    data = {
                        "question": question,
                        "user_answer": new_answer,
                        "correct_answer": result.correct_answer,
                        "score": result.score,
                        "feedback": result.feedback,
                        "explanation": result.explanation,
                        "knowledge_points": result.knowledge_points,
                        "detailed_analysis": result.detailed_analysis
                    }

                    self.db_manager.save_answer_record(self.current_user, data)

                    # 如果得分高于80，标记错题为已掌握
                    if result.score >= 80:
                        with self.db_manager._get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                UPDATE mistakes 
                                SET mastered = TRUE, updated_at = CURRENT_TIMESTAMP
                                WHERE user_id = ? AND question = ?
                            """, (self.current_user, question))
                            conn.commit()
                        print_color("\n✅ 错题已标记为已掌握！", Colors.GREEN)

                except Exception as e:
                    print_color(f"批改失败：{str(e)}", Colors.RED)

            input(f"\n{Colors.YELLOW}按回车键继续...{Colors.ENDC}")

        def _mark_mistake_mastered(self, mistakes):
            """标记错题为已掌握"""
            self.clear_screen()
            print_header("✅ 标记已掌握")

            print_color("请选择要标记为已掌握的错题编号（可多选，用逗号分隔），或输入0返回：", Colors.CYAN)

            # 显示错题
            display_count = min(10, len(mistakes))
            for i, mistake in enumerate(mistakes[:display_count], 1):
                question = mistake.get('question', '')
                if len(question) > 50:
                    question = question[:47] + "..."
                print_color(f"  [{i}] {question}", Colors.YELLOW)

            choice = input(f"\n{Colors.GREEN}选择: {Colors.ENDC}")

            if choice == '0':
                return

            try:
                # 解析多个选择
                choices = [int(x.strip()) for x in choice.split(',')]

                marked_count = 0
                with self.db_manager._get_connection() as conn:
                    cursor = conn.cursor()
                    for choice_num in choices:
                        if 1 <= choice_num <= display_count:
                            mistake = mistakes[choice_num - 1]
                            cursor.execute("""
                                UPDATE mistakes 
                                SET mastered = TRUE, updated_at = CURRENT_TIMESTAMP
                                WHERE user_id = ? AND question = ?
                            """, (self.current_user, mistake['question']))
                            marked_count += 1
                    conn.commit()

                if marked_count > 0:
                    print_color(f"\n✅ 成功标记 {marked_count} 道题目为已掌握！", Colors.GREEN)
                else:
                    print_color("没有标记任何题目", Colors.YELLOW)

            except ValueError:
                print_color("无效的输入", Colors.RED)

            time.sleep(1.5)

        def view_statistics(self):
            """查看学习统计"""
            self.clear_screen()
            print_header("📈 学习统计")

            profile = self.db_manager.get_user_profile(self.current_user)
            recent_scores = self.db_manager.get_recent_scores(self.current_user, limit=10)

            # 基本信息
            print_color("👤 用户信息：", Colors.CYAN)
            print_color(f"  用户名：{profile.get('name', 'N/A')} ({self.current_user})", Colors.YELLOW)
            print_color(f"  当前等级：{profile.get('level', 'N/A')}", Colors.YELLOW)

            # 添加等级进度条
            level = profile.get('level', '初级')
            avg_score = profile.get('avg_score', 0)
            if level == '初级':
                progress = min(100, (avg_score / 70) * 100)
                next_level = '中级'
            elif level == '中级':
                progress = min(100, ((avg_score - 70) / 15) * 100)
                next_level = '高级'
            else:
                progress = 100
                next_level = '最高级'

            if level != '高级':
                bar_length = 20
                filled = int(bar_length * progress / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                print_color(f"  升级进度：[{bar}] {progress:.0f}% → {next_level}", Colors.YELLOW)

            print()

            # 统计信息
            print_color("📊 学习统计：", Colors.CYAN)
            print_color(f"  总答题数：{profile.get('total_questions', 0)}", Colors.YELLOW)
            print_color(f"  正确数：{profile.get('correct_count', 0)}", Colors.YELLOW)
            print_color(f"  平均分：{profile.get('avg_score', 0):.1f}", Colors.YELLOW)

            if profile.get('total_questions', 0) > 0:
                correct_rate = profile.get('correct_count', 0) / profile.get('total_questions', 0) * 100

                # 根据正确率显示不同颜色
                if correct_rate >= 80:
                    rate_color = Colors.GREEN
                elif correct_rate >= 60:
                    rate_color = Colors.YELLOW
                else:
                    rate_color = Colors.RED

                print_color(f"  正确率：{rate_color}{correct_rate:.1f}%{Colors.ENDC}", Colors.YELLOW)
            print()

            # 最近成绩趋势
            if recent_scores:
                print_color("📅 最近成绩（最新10次）：", Colors.CYAN)

                # 计算趋势
                scores = [r['score'] for r in recent_scores]
                if len(scores) >= 2:
                    trend = scores[0] - scores[-1]  # 最新分数 - 最旧分数
                    if trend > 10:
                        trend_text = f"↑ 上升趋势 (+{trend:.0f}分)"
                        trend_color = Colors.GREEN
                    elif trend < -10:
                        trend_text = f"↓ 下降趋势 ({trend:.0f}分)"
                        trend_color = Colors.RED
                    else:
                        trend_text = "→ 保持稳定"
                        trend_color = Colors.YELLOW

                    print_color(f"  趋势：{trend_color}{trend_text}{Colors.ENDC}", Colors.YELLOW)

                # 显示成绩图表
                print()
                print_color("  成绩分布：", Colors.YELLOW)
                for i, record in enumerate(recent_scores, 1):
                    score = record.get('score', 0)
                    bar_length = int(score / 5)  # 每5分一个方块

                    if score >= 90:
                        bar_color = Colors.GREEN
                    elif score >= 70:
                        bar_color = Colors.YELLOW
                    else:
                        bar_color = Colors.RED

                    bar = '▪' * bar_length
                    time_str = record.get('created_at', '')[:10]
                    print(f"    {i:2d}. [{time_str}] {bar_color}{bar} {score}{Colors.ENDC}")
            else:
                print_color("📅 暂无答题记录", Colors.YELLOW)
            print()

            # 错题统计
            mistakes = self.db_manager.get_user_mistakes(self.current_user, limit=100)
            unmastered_count = len([m for m in mistakes if not m.get('mastered', False)])

            print_color("❌ 错题情况：", Colors.CYAN)
            print_color(f"  待复习错题：{unmastered_count} 道", Colors.YELLOW)
            if unmastered_count > 0:
                print_color("  💡 建议：定期复习错题，巩固薄弱知识点", Colors.YELLOW)
            print()

            # 学习建议
            print_color("🎯 个性化建议：", Colors.CYAN)
            if level == '初级':
                print_color("  💪 继续打好基础，多做基础练习", Colors.YELLOW)
                print_color("  📚 建议每天练习3-5道题", Colors.YELLOW)
                print_color("  🎯 目标：平均分达到70分升级到中级", Colors.YELLOW)
            elif level == '中级':
                print_color("  🔥 基础不错，可以挑战中等难度题目", Colors.YELLOW)
                print_color("  📚 建议尝试综合性题目", Colors.YELLOW)
                print_color("  🎯 目标：平均分达到85分升级到高级", Colors.YELLOW)
            else:
                print_color("  🎯 水平很高，可以挑战难题和综合题", Colors.YELLOW)
                print_color("  🏆 继续保持，你是学霸！", Colors.YELLOW)
                print_color("  💡 可以尝试帮助其他同学", Colors.YELLOW)

            # 学习会话统计
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as count, AVG(avg_score) as avg
                    FROM learning_sessions 
                    WHERE user_id = ? AND status = 'completed'
                """, (self.current_user,))
                session_stats = cursor.fetchone()

            if session_stats and session_stats['count'] > 0:
                print()
                print_color("🎓 学习会话统计：", Colors.CYAN)
                print_color(f"  完成会话数：{session_stats['count']}", Colors.YELLOW)
                print_color(f"  会话平均分：{session_stats['avg']:.1f}", Colors.YELLOW)

            input(f"\n{Colors.YELLOW}按回车键返回主菜单...{Colors.ENDC}")

        def view_history(self):
            """查看答题历史"""
            self.clear_screen()
            print_header("📜 答题历史")

            history = self.db_manager.get_answer_history(self.current_user, limit=20)

            if not history:
                print_color("暂无答题记录", Colors.YELLOW)
            else:
                print_color(f"最近 {len(history)} 条答题记录：", Colors.CYAN)
                print()

                for i, record in enumerate(history, 1):
                    score = record.get('score', 0)
                    if score >= 90:
                        score_color = Colors.GREEN
                        score_emoji = "✅"
                    elif score >= 60:
                        score_color = Colors.YELLOW
                        score_emoji = "⚠️ "
                    else:
                        score_color = Colors.RED
                        score_emoji = "❌"

                    time_str = record.get('created_at', '')[:19]
                    question = record.get('question', '')

                    if len(question) > 50:
                        question = question[:47] + "..."

                    print_color(f"{score_emoji} [{time_str}] {score_color}得分：{score:3d}{Colors.ENDC}", Colors.YELLOW)
                    print_color(f"   题目：{question}", Colors.YELLOW)

                    # 显示知识点
                    knowledge_str = record.get('knowledge_points', '[]')
                    try:
                        knowledge_points = json.loads(knowledge_str)
                        if knowledge_points:
                            print_color(f"   知识点：{', '.join(knowledge_points)}", Colors.CYAN)
                    except:
                        pass

                    # 如果有会话ID，显示会话信息
                    session_id = record.get('session_id')
                    if session_id:
                        print_color(f"   学习会话：{session_id[-8:]}", Colors.BLUE)

                    print()

            # 显示操作菜单
            if history:
                print_color("操作选项：", Colors.CYAN)
                print_color("  [v] 查看详细记录", Colors.GREEN)
                print_color("  [e] 导出历史记录", Colors.BLUE)
                print_color("  [回车] 返回主菜单", Colors.YELLOW)

                choice = input(f"\n{Colors.GREEN}请选择操作: {Colors.ENDC}").lower()

                if choice == 'v':
                    self._view_detailed_history(history)
                elif choice == 'e':
                    self._export_history(history)
            else:
                input(f"\n{Colors.YELLOW}按回车键返回主菜单...{Colors.ENDC}")

        def _view_detailed_history(self, history):
            """查看详细历史记录"""
            self.clear_screen()
            print_header("📋 详细历史记录")

            print_color("请输入要查看的记录编号（1-20），或输入0返回：", Colors.CYAN)

            try:
                choice = int(input(f"{Colors.GREEN}> {Colors.ENDC}"))
                if choice == 0:
                    return
                elif 1 <= choice <= len(history):
                    record = history[choice - 1]

                    self.clear_screen()
                    print_header("📋 答题记录详情")

                    print_color("基本信息：", Colors.CYAN)
                    print_color(f"  时间：{record.get('created_at', '')}", Colors.YELLOW)
                    print_color(f"  得分：{record.get('score', 0)}/100", Colors.YELLOW)
                    print()

                    print_color("题目：", Colors.CYAN)
                    print_color(f"  {record.get('question', '')}", Colors.YELLOW)
                    print()

                    print_color("你的答案：", Colors.CYAN)
                    print_color(f"  {record.get('user_answer', '')}", Colors.YELLOW)
                    print()

                    print_color("正确答案：", Colors.CYAN)
                    print_color(f"  {record.get('correct_answer', '')}", Colors.GREEN)
                    print()

                    print_color("反馈：", Colors.CYAN)
                    print_color(f"  {record.get('feedback', '')}", Colors.YELLOW)
                    print()

                    print_color("详细解析：", Colors.CYAN)
                    explanation = record.get('explanation', '')
                    for line in explanation.split('\n'):
                        print_color(f"  {line}", Colors.YELLOW)

                    input(f"\n{Colors.YELLOW}按回车键返回...{Colors.ENDC}")

            except ValueError:
                print_color("无效的输入", Colors.RED)
                time.sleep(1)

        def _export_history(self, history):
            """导出历史记录"""
            filename = f"history_{self.current_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join("data", filename)

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(history, f, ensure_ascii=False, indent=2, default=str)

                print_color(f"\n✅ 历史记录已导出到：{filepath}", Colors.GREEN)
            except Exception as e:
                print_color(f"\n❌ 导出失败：{str(e)}", Colors.RED)

            time.sleep(2)

        def switch_user(self):
            """切换用户"""
            self.clear_screen()
            print_header("🔄 切换用户")

            users = self.db_manager.get_all_users()

            if users:
                print_color("现有用户：", Colors.CYAN)
                for i, user in enumerate(users, 1):
                    # 显示用户信息和统计
                    level_color = {
                        '初级': Colors.GREEN,
                        '中级': Colors.YELLOW,
                        '高级': Colors.RED
                    }.get(user.get('level', ''), Colors.YELLOW)

                    print_color(f"  [{i}] {user.get('name')} ({user.get('user_id')})", Colors.CYAN)
                    print_color(f"      等级：{level_color}{user.get('level')}{Colors.ENDC} | " +
                                f"答题数：{user.get('total_questions', 0)} | " +
                                f"平均分：{user.get('avg_score', 0):.1f}", Colors.YELLOW)

                    # 高亮当前用户
                    if user.get('user_id') == self.current_user:
                        print_color("      ← 当前用户", Colors.GREEN)
                print()

            print_color("请选择：", Colors.CYAN)
            print_color("  [1-9] 选择现有用户", Colors.GREEN)
            print_color("  [n]   创建新用户", Colors.BLUE)
            print_color("  [d]   删除用户", Colors.RED)
            print_color("  [0]   取消", Colors.YELLOW)

            choice = input(f"\n{Colors.GREEN}请选择: {Colors.ENDC}").lower()

            if choice.isdigit() and users:
                user_num = int(choice)
                if 1 <= user_num <= len(users):
                    self.current_user = users[user_num - 1]['user_id']
                    print_color(f"\n✓ 已切换到用户：{users[user_num - 1]['name']} ({self.current_user})", Colors.GREEN)
                    time.sleep(1.5)
                elif user_num == 0:
                    return
                else:
                    print_color("无效的用户编号", Colors.RED)
                    time.sleep(1)

            elif choice == 'n':
                self._create_new_user()
            elif choice == 'd':
                self._delete_user(users)
            elif choice == '0':
                return
            else:
                print_color("无效的选择", Colors.RED)
                time.sleep(1)

        def _create_new_user(self):
            """创建新用户"""
            print()
            print_color("创建新用户", Colors.CYAN)
            print_color("请输入用户ID（英文字母和数字）：", Colors.CYAN)
            new_id = input(f"{Colors.YELLOW}> {Colors.ENDC}").strip()

            if not new_id:
                print_color("用户ID不能为空", Colors.RED)
                time.sleep(1)
                return

            # 检查ID是否已存在
            existing_users = self.db_manager.get_all_users()
            if any(u['user_id'] == new_id for u in existing_users):
                print_color("用户ID已存在", Colors.RED)
                time.sleep(1)
                return

            print_color("请输入用户名：", Colors.CYAN)
            new_name = input(f"{Colors.YELLOW}> {Colors.ENDC}").strip()

            if new_name:
                # 创建新用户
                self.db_manager.get_user_profile(new_id)  # 这会自动创建用户

                # 更新用户名
                with self.db_manager._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE users SET name = ? WHERE user_id = ?
                    """, (new_name, new_id))
                    conn.commit()

                self.current_user = new_id
                print_color(f"\n✓ 已创建并切换到用户：{new_name} ({new_id})", Colors.GREEN)
            else:
                print_color("用户名不能为空", Colors.RED)

            time.sleep(1.5)

        def _delete_user(self, users):
            """删除用户"""
            if not users:
                print_color("没有可删除的用户", Colors.YELLOW)
                time.sleep(1)
                return

            print()
            print_color("⚠️  警告：删除用户将清除所有相关数据！", Colors.RED)
            print_color("请输入要删除的用户编号，或输入0取消：", Colors.CYAN)

            try:
                choice = int(input(f"{Colors.RED}> {Colors.ENDC}"))
                if choice == 0:
                    return
                elif 1 <= choice <= len(users):
                    user_to_delete = users[choice - 1]

                    if user_to_delete['user_id'] == self.current_user:
                        print_color("不能删除当前用户", Colors.RED)
                        time.sleep(1)
                        return

                    # 确认删除
                    print_color(f"确认删除用户 {user_to_delete['name']} ({user_to_delete['user_id']})? (yes/no)",
                                Colors.RED)
                    confirm = input(f"{Colors.RED}> {Colors.ENDC}").lower()

                    if confirm == 'yes':
                        with self.db_manager._get_connection() as conn:
                            cursor = conn.cursor()
                            user_id = user_to_delete['user_id']

                            # 删除相关数据
                            cursor.execute("DELETE FROM answer_records WHERE user_id = ?", (user_id,))
                            cursor.execute("DELETE FROM mistakes WHERE user_id = ?", (user_id,))
                            cursor.execute("DELETE FROM learning_sessions WHERE user_id = ?", (user_id,))
                            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

                            conn.commit()

                        print_color(f"✓ 用户 {user_to_delete['name']} 已删除", Colors.GREEN)
                    else:
                        print_color("取消删除", Colors.YELLOW)
                else:
                    print_color("无效的选择", Colors.RED)
            except ValueError:
                print_color("无效的输入", Colors.RED)

            time.sleep(1.5)

        def system_info(self):
            """系统信息"""
            self.clear_screen()
            print_header("⚙️ 系统信息")

            # 数据库信息
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM answer_records")
                answer_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM mistakes")
                mistake_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM knowledge_points")
                knowledge_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM learning_sessions")
                session_count = cursor.fetchone()[0]

            # 向量数据库信息
            vector_count = len(self.vector_db.metadata)

            print_color("📊 系统状态：", Colors.CYAN)
            print_color(f"  ✅ 系统运行正常", Colors.GREEN)
            print_color(f"  🕐 当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.YELLOW)
            print()

            print_color("🗃️ 数据统计：", Colors.CYAN)
            print_color(f"  👥 用户数量：{user_count}", Colors.YELLOW)
            print_color(f"  📝 答题记录：{answer_count}", Colors.YELLOW)
            print_color(f"  ❌ 错题数量：{mistake_count}", Colors.YELLOW)
            print_color(f"  📚 知识点数：{knowledge_count}", Colors.YELLOW)
            print_color(f"  🎓 学习会话：{session_count}", Colors.YELLOW)
            print_color(f"  📊 向量文档：{vector_count}", Colors.YELLOW)
            print()

            print_color("🤖 AI配置：", Colors.CYAN)
            print_color(f"  API端点：{Config.OPENAI_BASE_URL}", Colors.YELLOW)
            print_color(f"  LLM模型：{Config.LLM_MODEL}", Colors.YELLOW)
            print_color(f"  嵌入模型：{Config.EMBEDDING_MODEL}", Colors.YELLOW)

            # 检查API连接
            print()
            print_color("🔌 API连接测试：", Colors.CYAN)
            try:
                response = self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                print_color(f"  ✅ API连接正常", Colors.GREEN)
            except:
                print_color(f"  ❌ API连接失败", Colors.RED)
            print()

            print_color("📁 存储路径：", Colors.CYAN)
            print_color(f"  数据库：{os.path.abspath(Config.DATABASE_PATH)}", Colors.YELLOW)
            print_color(f"  向量库：{os.path.abspath(Config.CHROMA_PATH)}", Colors.YELLOW)

            # 检查文件大小
            if os.path.exists(Config.DATABASE_PATH):
                db_size = os.path.getsize(Config.DATABASE_PATH) / 1024 / 1024  # MB
                print_color(f"  数据库大小：{db_size:.2f} MB", Colors.YELLOW)
            print()

            print_color("💻 系统环境：", Colors.CYAN)
            print_color(f"  Python版本：{sys.version.split()[0]}", Colors.YELLOW)
            print_color(f"  操作系统：{sys.platform}", Colors.YELLOW)
            print_color(f"  当前目录：{os.getcwd()}", Colors.YELLOW)

            input(f"\n{Colors.YELLOW}按回车键返回主菜单...{Colors.ENDC}")

# ============= 主程序 =============
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='智能教育Agent系统 - 命令行版本')
    parser.add_argument('--user', type=str, default='student_1', help='用户ID')
    parser.add_argument('--no-clear', action='store_true', help='不清屏')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    # 设置日志级别
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    try:
        # 打印启动信息
        print_color("🚀 正在启动智能教育Agent系统...", Colors.CYAN)

        # 初始化命令行界面
        cli = CommandLineInterface()
        cli.current_user = args.user

        print_color(f"✅ 系统初始化完成，当前用户：{args.user}", Colors.GREEN)
        time.sleep(1)

        # 启动主界面
        cli.show_main_menu()

    except KeyboardInterrupt:
        print_color("\n\n👋 感谢使用，再见！", Colors.GREEN)
        sys.exit(0)
    except Exception as e:
        print_color(f"❌ 系统错误：{str(e)}", Colors.RED)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

