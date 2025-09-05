"""结构化日志记录模块

提供完整的对话日志记录、错误追踪和问题排查功能。
"""

import json
import logging
import sqlite3
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """事件类型枚举"""
    DIALOGUE_TURN = "dialogue_turn"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    INTENT_RECOGNITION = "intent_recognition"
    API_CALL = "api_call"
    DEVICE_ACTION = "device_action"
    ERROR = "error"
    PERFORMANCE = "performance"
    FOCUS_SWITCH = "focus_switch"
    CLARIFICATION = "clarification"


@dataclass
class LogEntry:
    """日志条目数据结构"""
    
    log_id: str
    timestamp: float
    session_id: str
    event_type: str
    level: str
    message: str
    user_id: Optional[str] = None
    turn_id: Optional[int] = None
    processing_time: Optional[float] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None
    api_calls_count: Optional[int] = None
    error_type: Optional[str] = None
    error_traceback: Optional[str] = None
    context_data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)


class DialogueLogger:
    """对话系统结构化日志记录器"""
    
    def __init__(self, db_path: str = "data/dialogue_logs.db", max_log_age_days: int = 90):
        """初始化日志记录器
        
        Args:
            db_path: 日志数据库路径
            max_log_age_days: 日志保留天数
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_log_age_days = max_log_age_days
        
        # 设置标准日志器
        self.logger = logging.getLogger(__name__)
        self._setup_file_logging()
        
        # 初始化数据库
        self._init_database()
        
        # 性能统计
        self.performance_stats = {
            "total_turns": 0,
            "total_errors": 0,
            "avg_processing_time": 0.0,
            "api_call_count": 0
        }
    
    def _setup_file_logging(self):
        """设置文件日志记录"""
        log_dir = self.db_path.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # 创建文件处理器
        log_file = log_dir / f"dialogue_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        # 设置格式器
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 添加到日志器
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def _init_database(self):
        """初始化日志数据库"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 创建日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dialogue_logs (
                    log_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    user_id TEXT,
                    turn_id INTEGER,
                    processing_time REAL,
                    intent TEXT,
                    confidence REAL,
                    api_calls_count INTEGER,
                    error_type TEXT,
                    error_traceback TEXT,
                    context_data TEXT
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON dialogue_logs (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_session_id ON dialogue_logs (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_event_type ON dialogue_logs (event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_level ON dialogue_logs (level)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_user_id ON dialogue_logs (user_id)")
            
            # 创建性能监控表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    context_data TEXT
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON performance_metrics (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics (metric_name)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_log_id(self) -> str:
        """生成日志ID"""
        return f"log_{int(time.time() * 1000000)}"
    
    def _save_log_entry(self, entry: LogEntry):
        """保存日志条目到数据库"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO dialogue_logs (
                    log_id, timestamp, session_id, event_type, level, message,
                    user_id, turn_id, processing_time, intent, confidence,
                    api_calls_count, error_type, error_traceback, context_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.log_id,
                entry.timestamp,
                entry.session_id,
                entry.event_type,
                entry.level,
                entry.message,
                entry.user_id,
                entry.turn_id,
                entry.processing_time,
                entry.intent,
                entry.confidence,
                entry.api_calls_count,
                entry.error_type,
                entry.error_traceback,
                json.dumps(entry.context_data, ensure_ascii=False) if entry.context_data else None
            ))
            conn.commit()
    
    def log_dialogue_turn(self, session_id: str, turn_id: int, user_input: str, 
                         system_response: str, debug_info: Dict, user_id: str = None):
        """记录对话轮次日志
        
        Args:
            session_id: 会话ID
            turn_id: 轮次ID
            user_input: 用户输入
            system_response: 系统响应
            debug_info: 调试信息
            user_id: 用户ID
        """
        intent_result = debug_info.get("intent_result", {})
        processing_time = debug_info.get("processing_time", 0)
        api_calls = debug_info.get("api_calls", [])
        
        # 脱敏处理
        sanitized_input = self._sanitize_text(user_input)
        sanitized_response = self._sanitize_text(system_response)
        
        # 创建日志条目
        entry = LogEntry(
            log_id=self._generate_log_id(),
            timestamp=time.time(),
            session_id=session_id,
            event_type=EventType.DIALOGUE_TURN.value,
            level=LogLevel.INFO.value,
            message=f"对话轮次: 用户='{sanitized_input}', 系统='{sanitized_response}'",
            user_id=user_id,
            turn_id=turn_id,
            processing_time=processing_time,
            intent=intent_result.get("intent"),
            confidence=intent_result.get("confidence"),
            api_calls_count=len(api_calls),
            context_data={
                "intent_result": intent_result,
                "api_success_count": len([call for call in api_calls if call.get("success")]),
                "context_updates": debug_info.get("context_updates", {}),
                "device_action": debug_info.get("device_action_result", {}).get("message", ""),
                "clarification_needed": intent_result.get("need_clarification", False)
            }
        )
        
        self._save_log_entry(entry)
        
        # 同时记录到文件日志
        self.logger.info(
            f"TURN | Session:{session_id} | Turn:{turn_id} | "
            f"Intent:{intent_result.get('intent')} | "
            f"Confidence:{intent_result.get('confidence', 0):.2f} | "
            f"ProcessTime:{processing_time:.2f}s | "
            f"APIs:{len(api_calls)}"
        )
        
        # 更新性能统计
        self.performance_stats["total_turns"] += 1
        self.performance_stats["api_call_count"] += len(api_calls)
        if processing_time > 0:
            current_avg = self.performance_stats["avg_processing_time"]
            total_turns = self.performance_stats["total_turns"]
            self.performance_stats["avg_processing_time"] = (
                (current_avg * (total_turns - 1) + processing_time) / total_turns
            )
    
    def log_session_event(self, session_id: str, event_type: EventType, 
                         user_id: str = None, context: Dict = None):
        """记录会话事件日志
        
        Args:
            session_id: 会话ID
            event_type: 事件类型
            user_id: 用户ID
            context: 上下文信息
        """
        entry = LogEntry(
            log_id=self._generate_log_id(),
            timestamp=time.time(),
            session_id=session_id,
            event_type=event_type.value,
            level=LogLevel.INFO.value,
            message=f"会话事件: {event_type.value}",
            user_id=user_id,
            context_data=context
        )
        
        self._save_log_entry(entry)
        self.logger.info(f"SESSION | {event_type.value} | Session:{session_id} | User:{user_id}")
    
    def log_error(self, session_id: str, error: Exception, context: Dict = None, 
                  user_id: str = None, turn_id: int = None):
        """记录错误日志
        
        Args:
            session_id: 会话ID
            error: 异常对象
            context: 错误上下文
            user_id: 用户ID
            turn_id: 轮次ID
        """
        error_traceback = traceback.format_exc()
        
        entry = LogEntry(
            log_id=self._generate_log_id(),
            timestamp=time.time(),
            session_id=session_id,
            event_type=EventType.ERROR.value,
            level=LogLevel.ERROR.value,
            message=f"错误: {type(error).__name__}: {str(error)}",
            user_id=user_id,
            turn_id=turn_id,
            error_type=type(error).__name__,
            error_traceback=error_traceback,
            context_data=context
        )
        
        self._save_log_entry(entry)
        
        # 同时记录到文件日志
        self.logger.error(
            f"ERROR | Session:{session_id} | Turn:{turn_id} | "
            f"Type:{type(error).__name__} | Message:{str(error)}"
        )
        
        # 更新错误统计
        self.performance_stats["total_errors"] += 1
    
    def log_api_call(self, session_id: str, api_call_info: Dict, 
                     user_id: str = None, turn_id: int = None):
        """记录API调用日志
        
        Args:
            session_id: 会话ID
            api_call_info: API调用信息
            user_id: 用户ID
            turn_id: 轮次ID
        """
        success = api_call_info.get("success", False)
        response_time = api_call_info.get("response_time", 0)
        
        entry = LogEntry(
            log_id=self._generate_log_id(),
            timestamp=time.time(),
            session_id=session_id,
            event_type=EventType.API_CALL.value,
            level=LogLevel.INFO.value if success else LogLevel.WARNING.value,
            message=f"API调用: {'成功' if success else '失败'}, 耗时: {response_time:.2f}s",
            user_id=user_id,
            turn_id=turn_id,
            processing_time=response_time,
            context_data={
                "success": success,
                "error": api_call_info.get("error"),
                "model": api_call_info.get("request", {}).get("model"),
                "content_length": len(api_call_info.get("content", ""))
            }
        )
        
        self._save_log_entry(entry)
        
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"API | {status} | Session:{session_id} | Time:{response_time:.2f}s")
    
    def log_focus_switch(self, session_id: str, old_focus: str, new_focus: str, 
                        reason: str, user_id: str = None, turn_id: int = None):
        """记录焦点切换日志
        
        Args:
            session_id: 会话ID
            old_focus: 旧焦点
            new_focus: 新焦点
            reason: 切换原因
            user_id: 用户ID
            turn_id: 轮次ID
        """
        entry = LogEntry(
            log_id=self._generate_log_id(),
            timestamp=time.time(),
            session_id=session_id,
            event_type=EventType.FOCUS_SWITCH.value,
            level=LogLevel.INFO.value,
            message=f"焦点切换: {old_focus} -> {new_focus}, 原因: {reason}",
            user_id=user_id,
            turn_id=turn_id,
            context_data={
                "old_focus": old_focus,
                "new_focus": new_focus,
                "reason": reason
            }
        )
        
        self._save_log_entry(entry)
        self.logger.info(f"FOCUS | Session:{session_id} | {old_focus} -> {new_focus} | {reason}")
    
    def save_performance_metric(self, session_id: str, metric_name: str, 
                               metric_value: float, metric_unit: str = None, 
                               context: Dict = None):
        """保存性能指标
        
        Args:
            session_id: 会话ID
            metric_name: 指标名称
            metric_value: 指标值
            metric_unit: 指标单位
            context: 上下文信息
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics (
                    metric_id, timestamp, session_id, metric_name, 
                    metric_value, metric_unit, context_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"metric_{int(time.time() * 1000000)}",
                time.time(),
                session_id,
                metric_name,
                metric_value,
                metric_unit,
                json.dumps(context, ensure_ascii=False) if context else None
            ))
            conn.commit()
    
    def search_logs(self, session_id: str = None, event_type: str = None, 
                   level: str = None, start_time: float = None, 
                   end_time: float = None, keyword: str = None, 
                   limit: int = 100) -> List[LogEntry]:
        """搜索日志
        
        Args:
            session_id: 会话ID过滤
            event_type: 事件类型过滤
            level: 日志级别过滤
            start_time: 开始时间
            end_time: 结束时间
            keyword: 关键词搜索
            limit: 返回数量限制
            
        Returns:
            List[LogEntry]: 日志条目列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 构建查询条件
            conditions = []
            params = []
            
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            
            if event_type:
                conditions.append("event_type = ?")
                params.append(event_type)
            
            if level:
                conditions.append("level = ?")
                params.append(level)
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            if keyword:
                conditions.append("(message LIKE ? OR error_type LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
                SELECT * FROM dialogue_logs
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor.execute(query, params)
            
            logs = []
            for row in cursor.fetchall():
                context_data = json.loads(row[14]) if row[14] else None
                log_entry = LogEntry(
                    log_id=row[0],
                    timestamp=row[1],
                    session_id=row[2],
                    event_type=row[3],
                    level=row[4],
                    message=row[5],
                    user_id=row[6],
                    turn_id=row[7],
                    processing_time=row[8],
                    intent=row[9],
                    confidence=row[10],
                    api_calls_count=row[11],
                    error_type=row[12],
                    error_traceback=row[13],
                    context_data=context_data
                )
                logs.append(log_entry)
            
            return logs
    
    def get_session_summary(self, session_id: str) -> Dict:
        """获取会话摘要
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 会话摘要信息
        """
        logs = self.search_logs(session_id=session_id)
        
        if not logs:
            return {"error": "未找到会话日志"}
        
        # 统计信息
        total_turns = len([log for log in logs if log.event_type == EventType.DIALOGUE_TURN.value])
        total_errors = len([log for log in logs if log.level == LogLevel.ERROR.value])
        
        processing_times = [log.processing_time for log in logs if log.processing_time]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        api_calls = sum([log.api_calls_count for log in logs if log.api_calls_count]) or 0
        
        # 意图分布
        intents = [log.intent for log in logs if log.intent]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # 时间范围
        timestamps = [log.timestamp for log in logs]
        start_time = min(timestamps) if timestamps else 0
        end_time = max(timestamps) if timestamps else 0
        
        return {
            "session_id": session_id,
            "total_turns": total_turns,
            "total_errors": total_errors,
            "avg_processing_time": avg_processing_time,
            "total_api_calls": api_calls,
            "intent_distribution": intent_counts,
            "start_time": datetime.fromtimestamp(start_time).isoformat() if start_time else None,
            "end_time": datetime.fromtimestamp(end_time).isoformat() if end_time else None,
            "duration_seconds": end_time - start_time if start_time and end_time else 0
        }
    
    def export_logs(self, output_file: str, session_id: str = None, 
                   start_time: float = None, end_time: float = None):
        """导出日志到文件
        
        Args:
            output_file: 输出文件路径
            session_id: 会话ID过滤
            start_time: 开始时间
            end_time: 结束时间
        """
        logs = self.search_logs(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for log in logs:
                # 转换为JSON格式
                log_data = log.to_dict()
                log_data["timestamp_iso"] = datetime.fromtimestamp(log.timestamp).isoformat()
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        
        self.logger.info(f"导出了 {len(logs)} 条日志到 {output_file}")
    
    def cleanup_old_logs(self, days_to_keep: int = None):
        """清理旧日志
        
        Args:
            days_to_keep: 保留天数，默认使用初始化时的设置
        """
        if days_to_keep is None:
            days_to_keep = self.max_log_age_days
        
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 删除旧日志
            cursor.execute("DELETE FROM dialogue_logs WHERE timestamp < ?", (cutoff_time,))
            deleted_logs = cursor.rowcount
            
            # 删除旧性能指标
            cursor.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_time,))
            deleted_metrics = cursor.rowcount
            
            conn.commit()
        
        self.logger.info(f"清理了 {deleted_logs} 条日志和 {deleted_metrics} 条性能指标")
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息
        
        Returns:
            Dict: 性能统计数据
        """
        return self.performance_stats.copy()
    
    def _sanitize_text(self, text: str) -> str:
        """脱敏处理文本，移除敏感信息并压缩长文本
        
        Args:
            text: 原始文本
            
        Returns:
            str: 脱敏后的文本
        """
        if not text:
            return text
        
        # 先进行长文本压缩
        text = self._compress_long_text(text)
        
        # 脱敏处理
        import re
        
        # 手机号脱敏 (保留前3位和后4位)
        text = re.sub(r'(\d{3})\d{4}(\d{4})', r'\1****\2', text)
        
        # 身份证号脱敏 (保留前6位和后4位)
        text = re.sub(r'(\d{6})\d{8}(\d{4})', r'\1********\2', text)
        
        return text
    
    def _compress_long_text(self, text: str, max_length: int = 200) -> str:
        """压缩过长的文本，特别是重复字符
        
        Args:
            text: 原始文本
            max_length: 最大显示长度
            
        Returns:
            str: 压缩后的文本
        """
        if not text or len(text) <= max_length:
            return text
        
        # 快速检测：如果是简单重复字符，直接处理
        if len(set(text)) == 1:  # 全是同一个字符
            char = text[0]
            return f"{char}...(重复{len(text)}次)...{char}"
        
        # 快速检测：如果前100个字符都一样，可能是重复字符串
        if len(text) > 100:
            sample = text[:100]
            if len(set(sample)) <= 2:  # 最多2种不同字符
                import re
                # 使用更高效的正则匹配
                pattern = r'(.{1,10})\1{5,}'  # 匹配重复的短模式
                match = re.search(pattern, text[:200])  # 只检查前200个字符
                if match:
                    pattern_text = match.group(1)
                    total_matches = len(text) // len(pattern_text)
                    return f"{pattern_text}...(模式重复约{total_matches}次)"
        
        # 如果文本过长，直接截断
        prefix_len = max_length // 3
        suffix_len = max_length // 3
        middle_info = f"...(省略{len(text) - prefix_len - suffix_len}字符)..."
        return text[:prefix_len] + middle_info + text[-suffix_len:]


# 全局日志器实例
_dialogue_logger_instance = None


def get_dialogue_logger() -> DialogueLogger:
    """获取全局日志器实例"""
    global _dialogue_logger_instance
    if _dialogue_logger_instance is None:
        _dialogue_logger_instance = DialogueLogger()
    return _dialogue_logger_instance


def init_dialogue_logger(db_path: str = None, max_log_age_days: int = 90):
    """初始化全局日志器
    
    Args:
        db_path: 数据库路径
        max_log_age_days: 日志保留天数
    """
    global _dialogue_logger_instance
    _dialogue_logger_instance = DialogueLogger(
        db_path=db_path or "data/dialogue_logs.db",
        max_log_age_days=max_log_age_days
    )
