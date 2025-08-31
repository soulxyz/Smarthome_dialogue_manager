from typing import Dict, List, Optional, Tuple

import pytest

# 引入被测模块
from dialogue_manager import DialogueEngine, IntentRecognizer
from dialogue_manager.clarification import ClarificationAgent


class DummyAPIResponse:
    def __init__(
        self,
        success: bool,
        content: str = "",
        error_message: Optional[str] = None,
        model: str = "test-model",
        response_time: float = 0.01,
        raw_response: Optional[dict] = None,
        usage: Optional[dict] = None,
    ):
        self.success = success
        self.content = content
        self.error_message = error_message
        self.model = model
        self.response_time = response_time
        self.raw_response = raw_response or {}
        self.usage = usage or {}


class FakeAPIClient:
    def __init__(self, responses: List[DummyAPIResponse]):
        self._responses = list(responses)
        self.timeout = 30
        self.model_id = "test-model"

    def chat_completion(self, messages: List[Dict], max_tokens: Optional[int] = None, temperature: Optional[float] = None):
        # 弹出一个预置响应，若没有则返回默认成功响应
        if self._responses:
            return self._responses.pop(0)
        return DummyAPIResponse(success=True, content="好的")


class FakeMemoryManager:
    def __init__(self):
        self.saved_sessions: List[Tuple[str, List]] = []

    def load_user_context(self, user_id: str) -> Dict:
        return {"user_id": user_id}

    def get_session_patterns(self, session_id: str) -> Dict:
        return {}

    def save_session(self, session_id: str, dialogue_history: List, user_id: Optional[str] = None):
        self.saved_sessions.append((session_id, list(dialogue_history)))


@pytest.fixture()
def intent_recognizer():
    return IntentRecognizer()


@pytest.fixture()
def fake_api_success():
    # 返回一个成功的JSON候选
    return FakeAPIClient([DummyAPIResponse(success=True, content='["打开空调", "关闭电视"]')])


@pytest.fixture()
def fake_api_timeout():
    # 模拟一次超时失败
    return FakeAPIClient([DummyAPIResponse(success=False, content="", error_message="timeout")])


def test_intent_device_control_entities(intent_recognizer):
    result = intent_recognizer.recognize("打开客厅的灯", context={}, history=[])
    assert result["intent"] == "device_control"
    assert result["confidence"] >= 0.8
    # 设备与动作实体应被识别
    entity_types = {e["entity_type"] for e in result["entities"]}
    assert "device" in entity_types and "action" in entity_types
    # 设备值包含“灯”类别
    device_values = [e["value"] for e in result["entities"] if e["entity_type"] == "device"]
    assert "灯" in device_values


def test_intent_greeting(intent_recognizer):
    result = intent_recognizer.recognize("你好", context={}, history=[])
    assert result["intent"] == "greeting"
    assert result["confidence"] >= 0.9
    assert result["need_clarification"] is False


def test_intent_query_no_clarification(intent_recognizer):
    # 命中“现在.*温度”的查询模式，最低预期置信度>=0.7，且不需澄清
    result = intent_recognizer.recognize("现在卧室温度怎么样", context={}, history=[])
    assert result["intent"] == "query_status"
    assert 0.7 <= result["confidence"] <= 1.0
    assert result["need_clarification"] is False


def test_clarification_high_confidence(fake_api_success, intent_recognizer):
    agent = ClarificationAgent(api_client=fake_api_success, intent_recognizer=intent_recognizer, confidence_threshold=0.7)
    context = {"last_entities": {"devices": [{"value": "空调"}]}}
    result, candidates = agent.clarify("有点热", context, history=[])
    assert candidates, "应当返回候选列表"
    assert result is not None, "高置信度应直接返回结果"
    assert result.get("clarification_type") == "auto_high_confidence"
    assert result.get("intent") == "device_control"


def test_clarification_timeout_fallback(fake_api_timeout, intent_recognizer):
    agent = ClarificationAgent(api_client=fake_api_timeout, intent_recognizer=intent_recognizer, confidence_threshold=0.7)
    context = {"last_entities": {"devices": [{"value": "空调"}]}}
    result, candidates = agent.clarify("26度", context, history=[])
    # 超时会触发降级候选生成，应当至少有候选
    assert candidates, "超时降级策略应产生候选"
    # 降级候选如“调节空调温度到26度”通常可被识别为设备控制，置信度>=0.8
    # 因此很可能有结果返回
    assert result is not None
    assert result.get("intent") == "device_control"


def test_engine_process_input_basic(monkeypatch):
    # 构造引擎并替换其外部依赖
    engine = DialogueEngine(api_key="test")

    # 替换内存管理器，避免真实数据库I/O
    fake_mem = FakeMemoryManager()
    engine.memory_manager = fake_mem

    # 替换API Client（用于生成最终回复）
    api_client = FakeAPIClient([DummyAPIResponse(success=True, content="好的，已为您打开灯")])
    engine.api_client = api_client
    # ClarificationAgent 持有的 client 也需要同步
    engine.clarification_agent.api_client = api_client

    session_id = engine.start_session("user001")
    assert session_id.startswith("user001_")

    resp, debug_info = engine.process_input("打开卧室的灯")
    assert "好的" in resp
    assert debug_info["intent_result"]["intent"] == "device_control"
    assert engine.current_state.value == "listening"
    assert len(engine.dialogue_history) == 1
