"""测试夹具和工具函数

提供真实的、可重用的测试组件
"""

import tempfile
import shutil
import pytest
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from unittest.mock import patch

from dialogue_manager.engine import DialogueEngine, EngineConfig
from dialogue_manager.device_manager import DeviceManager
from dialogue_manager.memory import MemoryManager
from dialogue_manager.api_client import SiliconFlowClient
from dialogue_manager.intent import IntentRecognizer
from dialogue_manager.clarification import ClarificationAgent

from tests.config import get_test_config


@pytest.fixture(scope="session")
def test_config():
    """测试配置夹具"""
    return get_test_config()


@pytest.fixture(scope="session") 
def api_client(test_config):
    """真实API客户端夹具"""
    client = SiliconFlowClient(api_key=test_config.api_key)
    client.timeout = test_config.api_timeout
    client.max_retries = test_config.api_max_retries
    
    # 测试连接
    if test_config.use_real_api:
        try:
            is_connected = client.test_connection()
            if not is_connected:
                pytest.skip("API连接失败，跳过需要真实API的测试")
        except Exception as e:
            pytest.skip(f"API连接测试失败: {e}")
    
    return client


@pytest.fixture
def temp_db_path():
    """临时数据库路径夹具"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_path = tmp.name
    
    yield temp_path
    
    # 清理
    try:
        Path(temp_path).unlink(missing_ok=True)
        # 清理WAL和SHM文件
        Path(f"{temp_path}-wal").unlink(missing_ok=True)
        Path(f"{temp_path}-shm").unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def memory_manager(temp_db_path, test_config):
    """内存管理器夹具"""
    if test_config.use_in_memory_db:
        manager = MemoryManager(db_path=":memory:")
    else:
        manager = MemoryManager(db_path=temp_db_path)
    
    yield manager
    
    if test_config.cleanup_after_tests:
        try:
            manager.close()
        except Exception:
            pass


@pytest.fixture
def device_manager():
    """设备管理器夹具"""
    return DeviceManager()


@pytest.fixture
def engine_config(test_config):
    """引擎配置夹具"""
    return EngineConfig(
        max_turns=10,
        confidence_threshold=0.7,
        enable_device_manager=True,
        enable_context_entity_fill=True,
        focus_switch_policy="conservative",
        focus_entity_turn_decay=2,
        execution_mode="internal_first",
        always_record_api_traces=True
    )


@pytest.fixture
def dialogue_engine(api_client, engine_config, memory_manager, test_config):
    """对话引擎夹具"""
    engine = DialogueEngine(api_key=test_config.api_key, config=engine_config)
    
    # 替换内存管理器为测试专用的
    engine.memory_manager = memory_manager
    
    # 如果不使用真实API，可以选择性地mock一些方法
    if not test_config.use_real_api:
        # 保留API客户端但限制调用
        original_chat_completion = engine.api_client.chat_completion
        
        def limited_chat_completion(messages, **kwargs):
            # 对于测试，返回简化的响应
            if len(messages) > 0 and "test" in str(messages).lower():
                from dialogue_manager.api_client import APIResponse
                return APIResponse(
                    success=True,
                    content="测试响应：好的，我明白了。",
                    usage={"total_tokens": 10},
                    model="test-model",
                    response_time=0.1
                )
            return original_chat_completion(messages, **kwargs)
        
        engine.api_client.chat_completion = limited_chat_completion
    
    return engine


@pytest.fixture
def intent_recognizer(engine_config, device_manager):
    """意图识别器夹具"""
    return IntentRecognizer(
        config=engine_config,
        confidence_threshold=0.7,
        device_manager=device_manager
    )


@pytest.fixture
def clarification_agent(api_client, intent_recognizer, test_config):
    """澄清代理夹具"""
    return ClarificationAgent(
        api_client=api_client,
        intent_recognizer=intent_recognizer,
        confidence_threshold=0.7
    )


@pytest.fixture
def sample_user_inputs():
    """示例用户输入夹具"""
    return {
        "device_control": [
            "打开客厅的灯",
            "关闭主卧的空调", 
            "把电视音量调到50",
            "调高空调温度",
            "开启风扇"
        ],
        "query_status": [
            "客厅灯现在是什么状态",
            "空调温度是多少度",
            "电视开着吗",
            "风扇风速是几档"
        ],
        "greeting": [
            "你好",
            "早上好",
            "hi",
            "谢谢",
            "再见"
        ],
        "ambiguous": [
            "调节一下",
            "开一下",
            "太热了",
            "有点暗",
            "声音太大"
        ],
        "complex": [
            "打开客厅的灯，然后把空调温度调到26度",
            "关闭所有设备",
            "我要看电视，请打开电视和调暗灯光",
            "睡觉模式：关闭电视，调低风扇，关闭灯"
        ]
    }


@pytest.fixture
def dialogue_scenarios():
    """对话场景夹具"""
    return [
        {
            "name": "基础设备控制",
            "turns": [
                ("打开客厅的灯", "好的，已为您打开客厅的灯"),
                ("调节亮度到80%", "已将客厅灯亮度设置为80%"),
                ("关掉它", "好的，已为您关闭客厅的灯")
            ],
            "expected_focus_changes": ["灯", "灯", "灯"]
        },
        {
            "name": "焦点切换",
            "turns": [
                ("打开客厅的灯", "好的，已为您打开客厅的灯"),
                ("空调状态怎么样", "当前客厅空调已关闭"),
                ("打开空调", "好的，已为您打开客厅空调"),
                ("调高温度", "已将客厅空调温度调高")
            ],
            "expected_focus_changes": ["灯", "灯", "空调", "空调"]
        },
        {
            "name": "省略消解",
            "turns": [
                ("打开主卧的空调", "好的，已为您打开主卧的空调"),
                ("温度调到24度", "已将主卧空调温度设置为24度"),
                ("关掉", "好的，已为您关闭主卧的空调")
            ],
            "expected_focus_changes": ["空调", "空调", "空调"]
        }
    ]


@pytest.fixture
def performance_test_data():
    """性能测试数据夹具"""
    return {
        "concurrent_requests": [
            "打开客厅的灯",
            "关闭主卧的空调",
            "调节电视音量",
            "查询风扇状态",
            "设置空调温度"
        ] * 10,  # 50个并发请求
        "large_dialogue_history": []
    }
    
    # 生成更真实的大量对话历史
    realistic_conversations = [
        ("打开客厅的灯", "好的，已经为您打开客厅的灯", "device_control"),
        ("调节空调到22度", "已将空调温度设置为22度", "device_control"), 
        ("关闭电视", "电视已关闭", "device_control"),
        ("查询空调状态", "空调当前温度是24度，制冷模式", "query_status"),
        ("打开风扇", "风扇已打开，设置为中档", "device_control"),
        ("调高音量", "音量已调高", "device_control"),
        ("关掉台灯", "台灯已关闭", "device_control"),
        ("设置定时器", "请告诉我需要设置多长时间", "schedule_task"),
        ("查询今天天气", "今天多云，温度25-30度", "query_weather"),
        ("你好", "您好！有什么可以帮助您的吗？", "greeting")
    ]
    
    for i in range(100):
        user_input, system_response, intent = realistic_conversations[i % len(realistic_conversations)]
        fixture_data["large_dialogue_history"].append({
            "user_input": user_input,
            "system_response": system_response, 
            "intent": intent,
            "confidence": 0.8 + (i % 3) * 0.1,
            "timestamp": 1000000 + i * 60
        })
    
    return fixture_data


class APICallTracker:
    """API调用跟踪器"""
    
    def __init__(self):
        self.calls = []
        self.total_tokens = 0
        self.total_time = 0.0
        self.error_count = 0
    
    def track_call(self, messages, response, duration):
        """跟踪API调用"""
        call_info = {
            "messages": messages,
            "response": response,
            "duration": duration,
            "success": response.success if hasattr(response, 'success') else True,
            "tokens": response.usage.get('total_tokens', 0) if hasattr(response, 'usage') else 0
        }
        self.calls.append(call_info)
        
        if call_info["success"]:
            self.total_tokens += call_info["tokens"]
            self.total_time += duration
        else:
            self.error_count += 1
    
    def get_stats(self):
        """获取统计信息"""
        return {
            "total_calls": len(self.calls),
            "successful_calls": len(self.calls) - self.error_count,
            "error_count": self.error_count,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "average_time": self.total_time / max(1, len(self.calls) - self.error_count),
            "success_rate": (len(self.calls) - self.error_count) / max(1, len(self.calls))
        }


@pytest.fixture
def api_tracker():
    """API调用跟踪器夹具"""
    return APICallTracker()


def cleanup_test_files():
    """清理测试文件"""
    test_dirs = ["data/test_*", "temp_*", "test_*.db*"]
    for pattern in test_dirs:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def setup_logging():
    """设置测试日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置测试相关的日志级别
    logging.getLogger("dialogue_manager").setLevel(logging.DEBUG)
    logging.getLogger("tests").setLevel(logging.DEBUG)


# 性能测试相关的辅助函数
def measure_execution_time(func, *args, **kwargs):
    """测量函数执行时间"""
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


async def run_concurrent_tasks(tasks, max_concurrent=5):
    """并发运行任务"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[run_task(task) for task in tasks])


def create_test_dialogue_history(count=10):
    """创建测试对话历史"""
    import time
    history = []
    base_time = time.time()
    
    # 真实的对话样例
    realistic_examples = [
        ("打开灯", "灯已打开", "device_control"),
        ("关闭空调", "空调已关闭", "device_control"),
        ("查询温度", "当前温度26度", "query_status"),
        ("调节亮度", "亮度已调节", "device_control"),
        ("你好", "您好！", "greeting"),
        ("设置定时", "定时已设置", "schedule_task")
    ]
    
    for i in range(count):
        user_input, system_response, intent = realistic_examples[i % len(realistic_examples)]
        turn = {
            "turn_id": i + 1,
            "user_input": user_input,
            "system_response": system_response,
            "intent": intent,
            "confidence": 0.7 + (i % 4) * 0.1,
            "timestamp": base_time + i * 60,
            "context": {"test": True}
        }
        history.append(turn)
    
    return history
