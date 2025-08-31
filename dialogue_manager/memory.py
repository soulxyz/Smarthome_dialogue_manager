"""记忆管理模块

负责用户对话历史、上下文信息和偏好设置的存储与检索。
"""

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional


@dataclass
class UserProfile:
    """用户档案数据结构"""

    user_id: str
    preferences: Dict[str, Any]  # 用户偏好
    device_config: Dict[str, Any]  # 设备配置
    created_at: float
    updated_at: float


@dataclass
class SessionRecord:
    """会话记录数据结构"""

    session_id: str
    user_id: str
    start_time: float
    end_time: float
    turn_count: int
    context: Dict[str, Any]
    summary: str = ""


@dataclass
class DialogueRecord:
    """对话记录数据结构"""

    record_id: str
    session_id: str
    turn_id: int
    user_input: str
    system_response: str
    intent: str
    entities: List[Dict]
    confidence: float
    timestamp: float
    context_snapshot: Dict[str, Any]


class ConnectionPool:
    """SQLite连接池"""

    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self.pool = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        self.created_connections = 0

        # 预创建一些连接
        for _ in range(min(3, max_connections)):
            conn = self._create_connection()
            self.pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """创建新的数据库连接"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        self.created_connections += 1
        return conn

    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = None
        try:
            # 尝试从池中获取连接
            try:
                conn = self.pool.get_nowait()
            except Empty:
                # 池中没有可用连接，创建新连接
                with self.lock:
                    if self.created_connections < self.max_connections:
                        conn = self._create_connection()
                    else:
                        # 等待可用连接
                        conn = self.pool.get(timeout=5)

            yield conn

        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                try:
                    # 将连接放回池中
                    self.pool.put_nowait(conn)
                except:
                    # 池已满，关闭连接
                    conn.close()
                    with self.lock:
                        self.created_connections -= 1

    def close_all(self):
        """关闭所有连接"""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except Empty:
                break
        self.created_connections = 0


class MemoryManager:
    """记忆管理器"""

    def __init__(self, db_path: str = "data/dialogues.db", max_connections: int = 10):
        """初始化记忆管理器

        Args:
            db_path: 数据库文件路径
            max_connections: 最大连接数
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        # 会话级 PatternCache：{session_id: {intent: set(patterns)}}
        self.session_patterns: Dict[str, Dict[str, set]] = {}
        self.pool = ConnectionPool(str(self.db_path), max_connections)
        self._init_database()

    def _init_database(self):
        """初始化数据库表结构"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            # 用户档案表
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT NOT NULL,
                    device_config TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """
            )

            # 会话记录表
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    turn_count INTEGER DEFAULT 0,
                    context TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """
            )

            # 对话记录表
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dialogue_records (
                    record_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    user_input TEXT NOT NULL,
                    system_response TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    entities TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    context_snapshot TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """
            )

            # 创建索引以提高查询性能
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions (start_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dialogue_session_id ON dialogue_records (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dialogue_timestamp ON dialogue_records (timestamp)")

            conn.commit()
            self.logger.info("Database initialized successfully")

    def create_user_profile(self, user_id: str, preferences: Dict = None, device_config: Dict = None) -> UserProfile:
        """创建用户档案

        Args:
            user_id: 用户ID
            preferences: 用户偏好
            device_config: 设备配置

        Returns:
            UserProfile: 创建的用户档案
        """
        if preferences is None:
            preferences = {}
        if device_config is None:
            device_config = {}

        current_time = time.time()
        profile = UserProfile(
            user_id=user_id,
            preferences=preferences,
            device_config=device_config,
            created_at=current_time,
            updated_at=current_time,
        )

        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO user_profiles 
                (user_id, preferences, device_config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    profile.user_id,
                    json.dumps(profile.preferences, ensure_ascii=False),
                    json.dumps(profile.device_config, ensure_ascii=False),
                    profile.created_at,
                    profile.updated_at,
                ),
            )
            conn.commit()

        self.logger.info(f"Created user profile for {user_id}")
        return profile

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户档案

        Args:
            user_id: 用户ID

        Returns:
            Optional[UserProfile]: 用户档案，如果不存在则返回None
        """
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()

            if row:
                return UserProfile(
                    user_id=row[0],
                    preferences=json.loads(row[1]),
                    device_config=json.loads(row[2]),
                    created_at=row[3],
                    updated_at=row[4],
                )

        return None

    def update_user_preferences(self, user_id: str, preferences: Dict):
        """更新用户偏好

        Args:
            user_id: 用户ID
            preferences: 新的偏好设置
        """
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE user_profiles 
                SET preferences = ?, updated_at = ?
                WHERE user_id = ?
            """,
                (json.dumps(preferences, ensure_ascii=False), time.time(), user_id),
            )
            conn.commit()

        self.logger.info(f"Updated preferences for user {user_id}")

    def load_user_context(self, user_id: str) -> Dict:
        """加载用户上下文

        Args:
            user_id: 用户ID

        Returns:
            Dict: 用户上下文信息
        """
        context = {"user_id": user_id, "preferences": {}, "device_config": {}, "recent_intents": [], "recent_entities": []}

        # 加载用户档案
        profile = self.get_user_profile(user_id)
        if profile:
            context["preferences"] = profile.preferences
            context["device_config"] = profile.device_config

        # 加载最近的对话历史
        recent_records = self.get_recent_dialogues(user_id, limit=5)
        if recent_records:
            context["recent_intents"] = [record.intent for record in recent_records]
            context["recent_entities"] = [record.entities for record in recent_records]

        return context

    def save_session(self, session_id: str, dialogue_history: List, user_id: str = None):
        """保存会话记录

        Args:
            session_id: 会话ID
            dialogue_history: 对话历史列表
            user_id: 用户ID
        """
        if not dialogue_history:
            return

        # 从session_id中提取user_id（如果未提供）
        if user_id is None:
            user_id = session_id.split("_")[0]

        # 处理开始和结束时间
        if dialogue_history:
            first_turn = dialogue_history[0]
            last_turn = dialogue_history[-1]

            # 处理不同的数据格式
            if hasattr(first_turn, "timestamp"):  # DialogueTurn对象
                start_time = first_turn.timestamp
                end_time = last_turn.timestamp
            elif isinstance(first_turn, dict):  # 字典格式
                # 转换时间戳格式
                timestamp_str = first_turn.get("timestamp", "")
                if isinstance(timestamp_str, str) and ":" in timestamp_str:
                    from datetime import date, datetime

                    today = date.today()
                    dt = datetime.strptime(f"{today} {timestamp_str}", "%Y-%m-%d %H:%M:%S")
                    start_time = dt.timestamp()
                else:
                    start_time = time.time()

                timestamp_str = last_turn.get("timestamp", "")
                if isinstance(timestamp_str, str) and ":" in timestamp_str:
                    from datetime import date, datetime

                    today = date.today()
                    dt = datetime.strptime(f"{today} {timestamp_str}", "%Y-%m-%d %H:%M:%S")
                    end_time = dt.timestamp()
                else:
                    end_time = time.time()
            else:  # 其他格式，使用当前时间
                start_time = time.time()
                end_time = time.time()
        else:
            start_time = time.time()
            end_time = time.time()

        # 生成会话摘要
        summary = self._generate_session_summary(dialogue_history)

        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            # 保存会话记录
            cursor.execute(
                """
                INSERT OR REPLACE INTO sessions 
                (session_id, user_id, start_time, end_time, turn_count, context, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    user_id,
                    start_time,
                    end_time,
                    len(dialogue_history),
                    json.dumps({}, ensure_ascii=False),  # 会话级上下文
                    summary,
                ),
            )

            # 保存对话记录
            for i, turn in enumerate(dialogue_history):
                # 处理不同的数据格式（DialogueTurn对象或字典）
                if hasattr(turn, "turn_id"):  # DialogueTurn对象
                    turn_id = turn.turn_id
                    user_input = turn.user_input
                    system_response = turn.system_response
                    intent = turn.intent or "unknown"
                    confidence = turn.confidence
                    timestamp = turn.timestamp
                    context = turn.context or {}
                else:  # 字典格式（来自UI界面）
                    turn_id = i + 1
                    user_input = turn.get("user_input", "")
                    system_response = turn.get("system_response", "")
                    intent_result = turn.get("intent_result", {})
                    intent = intent_result.get("intent", "unknown")
                    confidence = intent_result.get("confidence", 0.0)
                    # 处理时间戳格式
                    timestamp_str = turn.get("timestamp", "")
                    if isinstance(timestamp_str, str) and ":" in timestamp_str:
                        # 转换 "HH:MM:SS" 格式为时间戳
                        from datetime import date, datetime

                        today = date.today()
                        dt = datetime.strptime(f"{today} {timestamp_str}", "%Y-%m-%d %H:%M:%S")
                        timestamp = dt.timestamp()
                    else:
                        timestamp = time.time()
                    context = turn.get("debug_info", {}).get("context_updates", {})

                record_id = f"{session_id}_{turn_id}"
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO dialogue_records 
                    (record_id, session_id, turn_id, user_input, system_response, 
                     intent, entities, confidence, timestamp, context_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record_id,
                        session_id,
                        turn_id,
                        user_input,
                        system_response,
                        intent,
                        json.dumps([], ensure_ascii=False),  # 实体列表
                        confidence,
                        timestamp,
                        json.dumps(context, ensure_ascii=False),
                    ),
                )

            conn.commit()

        self.logger.info(f"Saved session {session_id} with {len(dialogue_history)} turns")

    def get_recent_dialogues(self, user_id: str, limit: int = 10) -> List[DialogueRecord]:
        """获取用户最近的对话记录

        Args:
            user_id: 用户ID
            limit: 返回记录数量限制

        Returns:
            List[DialogueRecord]: 对话记录列表
        """
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT dr.* FROM dialogue_records dr
                JOIN sessions s ON dr.session_id = s.session_id
                WHERE s.user_id = ?
                ORDER BY dr.timestamp DESC
                LIMIT ?
            """,
                (user_id, limit),
            )

            records = []
            for row in cursor.fetchall():
                record = DialogueRecord(
                    record_id=row[0],
                    session_id=row[1],
                    turn_id=row[2],
                    user_input=row[3],
                    system_response=row[4],
                    intent=row[5],
                    entities=json.loads(row[6]),
                    confidence=row[7],
                    timestamp=row[8],
                    context_snapshot=json.loads(row[9]),
                )
                records.append(record)

            return records

    def search_dialogues(self, user_id: str, keyword: str, limit: int = 20) -> List[DialogueRecord]:
        """搜索对话记录

        Args:
            user_id: 用户ID
            keyword: 搜索关键词
            limit: 返回记录数量限制

        Returns:
            List[DialogueRecord]: 匹配的对话记录列表
        """
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT dr.* FROM dialogue_records dr
                JOIN sessions s ON dr.session_id = s.session_id
                WHERE s.user_id = ? AND (
                    dr.user_input LIKE ? OR 
                    dr.system_response LIKE ? OR
                    dr.intent LIKE ?
                )
                ORDER BY dr.timestamp DESC
                LIMIT ?
            """,
                (user_id, f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit),
            )

            records = []
            for row in cursor.fetchall():
                record = DialogueRecord(
                    record_id=row[0],
                    session_id=row[1],
                    turn_id=row[2],
                    user_input=row[3],
                    system_response=row[4],
                    intent=row[5],
                    entities=json.loads(row[6]),
                    confidence=row[7],
                    timestamp=row[8],
                    context_snapshot=json.loads(row[9]),
                )
                records.append(record)

            return records

    def _generate_session_summary(self, dialogue_history: List) -> str:
        """生成会话摘要

        Args:
            dialogue_history: 对话历史

        Returns:
            str: 会话摘要
        """
        if not dialogue_history:
            return "空会话"

        # 统计意图分布
        intents = []
        for turn in dialogue_history:
            if hasattr(turn, "intent"):  # DialogueTurn对象
                if turn.intent:
                    intents.append(turn.intent)
            elif isinstance(turn, dict):  # 字典格式
                intent_result = turn.get("intent_result", {})
                intent = intent_result.get("intent")
                if intent:
                    intents.append(intent)
            # 其他格式跳过

        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # 生成简单摘要
        main_intent = max(intent_counts, key=intent_counts.get) if intent_counts else "unknown"
        summary = f"共{len(dialogue_history)}轮对话，主要意图：{main_intent}"

        return summary

    def get_statistics(self, user_id: str = None) -> Dict:
        """获取统计信息

        Args:
            user_id: 用户ID，如果为None则返回全局统计

        Returns:
            Dict: 统计信息
        """
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            if user_id:
                # 用户级统计
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM sessions s WHERE s.user_id = ?
                """,
                    (user_id,),
                )
                stats["total_sessions"] = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT COUNT(*) FROM dialogue_records dr
                    JOIN sessions s ON dr.session_id = s.session_id
                    WHERE s.user_id = ?
                """,
                    (user_id,),
                )
                stats["total_turns"] = cursor.fetchone()[0]
            else:
                # 全局统计
                cursor.execute("SELECT COUNT(*) FROM sessions")
                stats["total_sessions"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM dialogue_records")
                stats["total_turns"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM user_profiles")
                stats["total_users"] = cursor.fetchone()[0]

            return stats

    def cleanup_old_records(self, days_to_keep: int = 30):
        """清理旧记录

        Args:
            days_to_keep: 保留天数
        """
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)

        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            # 删除旧的对话记录
            cursor.execute("DELETE FROM dialogue_records WHERE timestamp < ?", (cutoff_time,))

            # 删除旧的会话记录
            cursor.execute("DELETE FROM sessions WHERE start_time < ?", (cutoff_time,))

            conn.commit()

        self.logger.info(f"Cleaned up records older than {days_to_keep} days")

    def close(self):
        """关闭内存管理器，清理资源"""
        self.pool.close_all()
        self.logger.info("MemoryManager closed")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    # ====== Session PatternCache APIs ======
    def add_pattern(self, session_id: str, intent: str, pattern: str):
        """向当前会话缓存中添加新正则模式"""
        if not pattern:
            return
        self.session_patterns.setdefault(session_id, {}).setdefault(intent, set()).add(pattern)

    def get_session_patterns(self, session_id: str) -> Dict[str, set]:
        """获取指定会话的所有自学习模式"""
        return self.session_patterns.get(session_id, {})

    def clear_session_patterns(self, session_id: str):
        """清理会话结束时的 PatternCache"""
        if session_id in self.session_patterns:
            del self.session_patterns[session_id]
