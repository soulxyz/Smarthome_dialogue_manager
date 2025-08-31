# Smart Home Dialogue Management System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

智能家居多轮对话管理引擎，基于大语言模型的自然语言处理系统。

## 项目简介

本系统为智能家居场景设计的对话管理解决方案，集成硅基流动API，提供意图识别、实体抽取、多轮对话管理和智能澄清功能。采用模块化架构，支持生产环境部署。

## 核心功能

- 多轮对话状态管理和会话跟踪
- 基于正则表达式和上下文的混合意图识别
- 设备、动作、数值等实体类型抽取
- 基于LLM的低置信度意图澄清机制
- SQLite持久化存储用户偏好和对话历史
- Streamlit调试监控界面
- API调用统计和性能监控
- 错误处理和降级策略

## 技术架构

### 系统组件

```
dialogue_manager/
├── engine.py          # 对话引擎 - 会话管理和流程控制
├── intent.py          # 意图识别器 - 混合模式意图分类
├── memory.py          # 记忆管理器 - 数据持久化
├── api_client.py      # API客户端 - 硅基流动API集成
└── clarification.py   # 澄清代理 - LLM智能澄清
```

### 数据流

1. 用户输入 → 对话引擎
2. 意图识别 → 实体抽取
3. 置信度评估 → 澄清判断
4. 响应生成 → 状态更新
5. 历史记录 → 持久化存储

## 安装部署

### 环境要求

- Python 3.8+
- 2GB+ 内存
- 稳定网络连接

### 安装步骤

```bash
# 克隆项目
git clone <repository-url>
cd smart-home-dialogue

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置

创建 `.env` 文件：

```bash
SILICONFLOW_API_KEY=your_api_key_here
DATABASE_PATH=./data/dialogues.db
LOG_LEVEL=INFO
```

或配置 Streamlit secrets：

```bash
mkdir .streamlit
echo 'SILICONFLOW_API_KEY = "your_api_key"' > .streamlit/secrets.toml
```

### 启动服务

```bash
# 调试界面
streamlit run ui/app.py --server.port=8503

# 或使用 Makefile
make dev
```

## API使用

### 基本用法

```python
from dialogue_manager import DialogueEngine
from dialogue_manager.engine import EngineConfig

# 配置引擎
config = EngineConfig(
    max_turns=10,
    confidence_threshold=0.7,
    model_name="deepseek-chat"
)

# 初始化
engine = DialogueEngine(api_key, config)

# 开始会话
session_id = engine.start_session("user_001")

# 处理输入
response, debug_info = engine.process_input("打开客厅的灯")

# 结束会话
engine.end_session()
```

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| max_turns | int | 10 | 最大对话轮数 |
| confidence_threshold | float | 0.7 | 置信度阈值 |
| model_name | str | "deepseek-chat" | LLM模型名称 |
| enable_clarification | bool | True | 启用智能澄清 |
| session_timeout | int | 3600 | 会话超时时间(秒) |

## 开发测试

### 运行测试

```bash
# 单元测试
pytest tests/ -v

# 覆盖率测试
pytest --cov=dialogue_manager tests/

# 代码检查
flake8 dialogue_manager/
mypy dialogue_manager/

# 格式化
black .
isort .
```

### 使用 Makefile

```bash
make test          # 运行测试
make lint          # 代码检查
make format        # 代码格式化
make ci            # 完整CI流程
```

## 性能指标

- 意图识别准确率: 85%+
- 平均响应时间: <2秒
- API调用成功率: 95%+
- 澄清成功率: 80%+

## 部署配置

### 生产环境

```bash
# 使用 gunicorn 部署
pip install gunicorn
gunicorn --bind 0.0.0.0:8000 --workers 4 app:app
```

### Docker 部署

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8503
CMD ["streamlit", "run", "ui/app.py", "--server.port=8503", "--server.address=0.0.0.0"]
```

## 故障排除

### 常见问题

1. **API密钥错误**
   - 检查环境变量配置
   - 验证API密钥有效性

2. **数据库连接失败**
   - 确认数据库文件路径
   - 检查文件权限

3. **模型调用超时**
   - 检查网络连接
   - 调整超时参数

### 日志配置

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dialogue_manager')
```

## 安全考虑

- API密钥通过环境变量管理
- 输入参数验证和清理
- SQL注入防护（参数化查询）
- 错误信息脱敏处理

## 许可证

MIT License

## 技术支持

- 提交 Issue 报告问题
- 查看项目文档和代码注释
- 使用调试界面进行问题诊断