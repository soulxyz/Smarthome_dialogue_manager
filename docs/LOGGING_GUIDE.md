# 对话日志系统使用指南

本指南介绍如何使用对话系统的完整日志记录功能，包括日志查看、问题排查和性能分析。

## 📋 功能概述

新的日志系统提供以下功能：

### 🎯 核心功能
- **结构化日志记录**: 自动记录所有对话轮次、API调用、错误和性能指标
- **智能日志搜索**: 支持按时间、会话、事件类型、日志级别等多维度搜索
- **实时监控**: 在UI界面中实时查看日志和系统状态
- **错误追踪**: 详细记录错误堆栈和上下文信息
- **性能分析**: 统计处理时间、API调用次数等性能指标
- **日志导出**: 支持将日志导出为JSON格式文件
- **自动清理**: 定期清理过期日志，节省存储空间

### 🔍 日志类型
- `dialogue_turn`: 对话轮次记录
- `session_start/end`: 会话开始/结束事件
- `intent_recognition`: 意图识别详情
- `api_call`: API调用记录
- `device_action`: 设备操作日志
- `focus_switch`: 焦点实体切换记录
- `error`: 错误和异常记录
- `performance`: 性能指标记录

## 🖥️ UI界面使用

### 访问日志面板
1. 启动Streamlit应用: `streamlit run ui/app.py`
2. 打开浏览器访问应用
3. 点击 **"📋 日志"** 标签页

### 日志搜索
在日志面板中，你可以：

1. **按会话搜索**: 输入会话ID查看特定会话的所有日志
2. **按事件类型过滤**: 选择要查看的事件类型（对话轮次、API调用等）
3. **按日志级别过滤**: 筛选ERROR、WARNING、INFO等不同级别的日志
4. **关键词搜索**: 在日志消息中搜索特定关键词
5. **时间范围过滤**: 选择最近1小时、24小时、7天或自定义时间范围

### 详细日志查看
- 搜索结果以表格形式展示，包含时间、级别、事件类型等关键信息
- 点击选择特定日志条目查看详细信息
- 查看错误堆栈、上下文数据、性能指标等详细内容

### 日志管理
- **会话摘要**: 查看特定会话的统计信息和意图分布
- **导出日志**: 将搜索结果导出为JSON文件
- **清理日志**: 删除超过指定天数的旧日志

## 💻 命令行工具使用

系统提供了 `scripts/debug_logs.py` 命令行工具，用于高级日志分析和问题排查。

### 安装和设置
```bash
cd project2
python scripts/debug_logs.py --help
```

### 主要命令

#### 1. 分析特定会话
```bash
python scripts/debug_logs.py session <session_id>
```
输出会话的完整分析报告，包括：
- 会话摘要（总轮数、错误数、平均处理时间等）
- 详细日志时间线
- 意图分布分析

#### 2. 查找错误日志
```bash
# 查找最近24小时的错误
python scripts/debug_logs.py errors

# 查找最近48小时的错误，最多返回100条
python scripts/debug_logs.py errors --hours 48 --limit 100
```
输出：
- 按错误类型分组的错误统计
- 每种错误的发生时间和上下文
- 错误趋势分析

#### 3. 性能分析
```bash
# 分析最近24小时的性能
python scripts/debug_logs.py performance

# 分析最近一周的性能
python scripts/debug_logs.py performance --hours 168
```
输出：
- 对话处理性能统计
- API调用成功率和响应时间
- 意图识别置信度分析
- 慢请求识别

#### 4. 导出会话报告
```bash
# 导出会话报告到默认文件
python scripts/debug_logs.py report <session_id>

# 导出到指定文件
python scripts/debug_logs.py report <session_id> --output report.json
```

#### 5. 实时监控
```bash
# 每30秒刷新一次
python scripts/debug_logs.py monitor

# 每10秒刷新一次
python scripts/debug_logs.py monitor --interval 10
```

## 🏗️ 开发者集成

### 在代码中使用日志系统

#### 基本使用
```python
from dialogue_manager.logger import get_dialogue_logger, EventType

logger = get_dialogue_logger()

# 记录对话轮次
logger.log_dialogue_turn(
    session_id="session_123",
    turn_id=1,
    user_input="打开客厅的灯",
    system_response="好的，已为您打开客厅的灯",
    debug_info={
        "intent_result": {"intent": "device_control", "confidence": 0.95},
        "processing_time": 1.2,
        "api_calls": []
    },
    user_id="user_456"
)

# 记录错误
try:
    # 一些可能出错的操作
    pass
except Exception as e:
    logger.log_error(
        session_id="session_123",
        error=e,
        context={"operation": "device_control", "device": "light"},
        user_id="user_456",
        turn_id=1
    )

# 记录焦点切换
logger.log_focus_switch(
    session_id="session_123",
    old_focus="灯",
    new_focus="空调",
    reason="用户明确提及新设备",
    user_id="user_456",
    turn_id=2
)
```

#### 性能指标记录
```python
# 记录自定义性能指标
logger.save_performance_metric(
    session_id="session_123",
    metric_name="device_response_time",
    metric_value=0.5,
    metric_unit="seconds",
    context={"device_type": "light", "room": "living_room"}
)
```

### 日志配置
日志系统支持以下配置：

```python
from dialogue_manager.logger import init_dialogue_logger

# 初始化自定义日志配置
init_dialogue_logger(
    db_path="custom/path/dialogue_logs.db",  # 自定义数据库路径
    max_log_age_days=60  # 日志保留60天
)
```

## 📊 数据结构

### 日志条目结构
```python
@dataclass
class LogEntry:
    log_id: str                     # 唯一日志ID
    timestamp: float               # 时间戳
    session_id: str               # 会话ID
    event_type: str               # 事件类型
    level: str                    # 日志级别
    message: str                  # 日志消息
    user_id: Optional[str]        # 用户ID
    turn_id: Optional[int]        # 轮次ID
    processing_time: Optional[float]  # 处理时间
    intent: Optional[str]         # 识别意图
    confidence: Optional[float]   # 置信度
    api_calls_count: Optional[int]   # API调用次数
    error_type: Optional[str]     # 错误类型
    error_traceback: Optional[str]   # 错误堆栈
    context_data: Optional[Dict]  # 上下文数据
```

### 数据库表结构

#### dialogue_logs 表
- `log_id`: 主键，日志唯一标识
- `timestamp`: 时间戳索引
- `session_id`: 会话ID索引
- `event_type`: 事件类型索引
- `level`: 日志级别索引
- 其他字段存储详细日志信息

#### performance_metrics 表
- `metric_id`: 主键，指标唯一标识
- `timestamp`: 时间戳索引
- `session_id`: 会话ID索引
- `metric_name`: 指标名称索引
- `metric_value`: 指标值
- `context_data`: 上下文信息JSON

## 🔧 维护和优化

### 定期维护
1. **日志清理**: 定期运行清理命令删除过期日志
```python
logger.cleanup_old_logs(days_to_keep=90)
```

2. **数据库优化**: SQLite数据库会自动进行WAL模式优化，支持并发读写

3. **存储监控**: 监控日志文件大小，必要时调整保留策略

### 性能优化
- 日志写入使用连接池，支持高并发
- 索引优化，快速搜索和过滤
- 异步写入，不影响主业务流程
- 自动批量提交，提高写入效率

### 故障排查
1. **日志写入失败**: 检查数据库文件权限和磁盘空间
2. **搜索性能慢**: 检查索引状态，考虑清理旧数据
3. **内存占用高**: 调整连接池大小和缓存策略

## 📈 最佳实践

### 1. 日志级别使用
- `DEBUG`: 开发调试信息
- `INFO`: 正常业务流程记录
- `WARNING`: 可能的问题，但不影响功能
- `ERROR`: 错误和异常
- `CRITICAL`: 严重错误，影响系统运行

### 2. 敏感信息保护
- 系统自动对手机号、身份证等敏感信息进行脱敏
- 避免在日志中记录密码、密钥等机密信息
- 上下文数据不应包含用户隐私信息

### 3. 性能考虑
- 避免记录过大的上下文数据
- 合理设置日志保留期限
- 定期进行数据库维护和优化

### 4. 监控告警
- 监控错误日志数量异常增长
- 关注API调用成功率下降
- 跟踪平均处理时间变化

## 🔗 相关文档

- [系统架构文档](./dialogue_system_improvement_guide.md)
- [API客户端文档](../dialogue_manager/api_client.py)
- [意图识别文档](../dialogue_manager/intent.py)
- [引擎核心文档](../dialogue_manager/engine.py)

## 🆘 问题反馈

如果在使用日志系统时遇到问题，请：

1. 首先查看错误日志: `python scripts/debug_logs.py errors`
2. 检查系统性能: `python scripts/debug_logs.py performance`
3. 导出相关会话报告用于问题分析
4. 联系开发团队并提供详细的错误信息和上下文

---

*最后更新: 2024年12月19日*
