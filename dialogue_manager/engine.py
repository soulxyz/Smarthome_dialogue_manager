"""对话引擎核心模块

负责多轮对话状态管理、会话流程控制和指令生成。
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .intent import IntentRecognizer
from .memory import MemoryManager
from .api_client import SiliconFlowClient
from .clarification import ClarificationAgent


@dataclass
class EngineConfig:
    """对话引擎配置类"""

    max_turns: int = 10
    confidence_threshold: float = 0.7
    model_name: str = "deepseek-chat"
    enable_clarification: bool = True
    session_timeout: int = 3600  # 会话超时时间（秒）

    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class DialogueState(Enum):
    """对话状态枚举"""

    IDLE = "idle"  # 空闲状态
    LISTENING = "listening"  # 监听用户输入
    PROCESSING = "processing"  # 处理中
    CLARIFYING = "clarifying"  # 澄清中
    EXECUTING = "executing"  # 执行指令
    ERROR = "error"  # 错误状态


@dataclass
class DialogueTurn:
    """单轮对话数据结构"""

    turn_id: int
    user_input: str
    system_response: str
    intent: Optional[str] = None
    confidence: float = 0.0
    timestamp: float = 0.0
    context: Dict = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.context is None:
            self.context = {}


class DialogueEngine:
    """对话引擎主类"""

    def __init__(self, api_key: str, config: Optional[EngineConfig] = None):
        """初始化对话引擎

        Args:
            api_key: 硅基流动API密钥
            config: 引擎配置对象
        """
        self.config = config or EngineConfig()
        self.api_client = SiliconFlowClient(api_key)
        self.intent_recognizer = IntentRecognizer(confidence_threshold=self.config.confidence_threshold)
        self.memory_manager = MemoryManager()
        self.clarification_agent = ClarificationAgent(
            self.api_client, self.intent_recognizer, confidence_threshold=self.config.confidence_threshold
        )

        self.current_state = DialogueState.IDLE
        self.session_id = None
        self.dialogue_history: List[DialogueTurn] = []
        self.context = {}

        self.logger = logging.getLogger(__name__)

    def update_config(self, **kwargs):
        """更新引擎配置"""
        self.config.update(**kwargs)
        self.logger.info(f"Engine config updated: {kwargs}")

    def start_session(self, user_id: str) -> str:
        """开始新的对话会话

        Args:
            user_id: 用户ID

        Returns:
            session_id: 会话ID
        """
        self.session_id = f"{user_id}_{int(time.time())}"
        self.current_state = DialogueState.LISTENING
        self.dialogue_history.clear()
        self.context = self.memory_manager.load_user_context(user_id)
        # 注入会话级动态模式
        self.intent_recognizer.set_session_patterns(self.memory_manager.get_session_patterns(self.session_id))

        self.logger.info(f"Started new session: {self.session_id}")
        return self.session_id

    def process_input(self, user_input: str) -> Tuple[str, Dict]:
        """处理用户输入

        Args:
            user_input: 用户输入文本

        Returns:
            Tuple[response, debug_info]: 系统响应和调试信息
        """
        start_time = time.time()
        debug_info = {
            "input": user_input,
            "processing_time": 0,
            "state_transitions": [],
            "intent_result": {},
            "api_calls": [],
            "context_updates": {},
        }

        try:
            # 状态转换：监听 -> 处理中
            self._transition_state(DialogueState.PROCESSING, debug_info)

            # 检查是否有待确认的指令
            if self._check_user_confirmation(user_input, debug_info):
                # 用户确认了待执行的指令
                pending_result = self.context.get("pending_confirmation")
                if pending_result:
                    debug_info["intent_result"] = pending_result
                    self._update_context(pending_result, debug_info)
                    # 清除待确认状态和澄清轮次
                    self.context.pop("pending_confirmation", None)
                    self.context.pop("clarification_rounds", None)
                    response = self._generate_response(pending_result, debug_info)
                    self._transition_state(DialogueState.EXECUTING, debug_info)
                else:
                    response = "好的，我明白了。还有什么可以帮您的吗？"
                    # 清除澄清轮次
                    self.context.pop("clarification_rounds", None)
                    self._transition_state(DialogueState.LISTENING, debug_info)
            else:
                # 正常的意图识别流程
                intent_result = self.intent_recognizer.recognize(user_input, self.context, self.dialogue_history)
                debug_info["intent_result"] = intent_result

                # 判断是否需要澄清
                if intent_result["confidence"] < self.config.confidence_threshold:  # 置信度阈值
                    # 检查澄清轮次，避免重复澄清循环
                    clarification_rounds = self.context.get("clarification_rounds", 0)
                    if clarification_rounds >= 2:  # 最多澄清2轮
                        response = (
                            "抱歉，我还是没有理解您的意思。请您换个方式描述，或者直接告诉我您想要控制哪个设备做什么操作。"
                        )
                        # 重置澄清轮次
                        self.context["clarification_rounds"] = 0
                        self._transition_state(DialogueState.LISTENING, debug_info)
                    else:
                        # 增加澄清轮次
                        self.context["clarification_rounds"] = clarification_rounds + 1
                        response = self._handle_clarification(intent_result, debug_info)
                        self._transition_state(DialogueState.CLARIFYING, debug_info)
                else:
                    # 成功识别意图，清除澄清轮次
                    self.context.pop("clarification_rounds", None)
                    response = self._generate_response(intent_result, debug_info)
                    self._update_context(intent_result, debug_info)
                    self._transition_state(DialogueState.EXECUTING, debug_info)

            # 记录对话轮次
            turn = DialogueTurn(
                turn_id=len(self.dialogue_history) + 1,
                user_input=user_input,
                system_response=response,
                intent=intent_result.get("intent"),
                confidence=intent_result.get("confidence", 0.0),
                context=self.context.copy(),
            )
            self.dialogue_history.append(turn)

            # 更新上下文
            self._update_context(intent_result, debug_info)

            # 状态转换：执行 -> 监听
            self._transition_state(DialogueState.LISTENING, debug_info)

        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            self._transition_state(DialogueState.ERROR, debug_info)
            response = "抱歉，系统出现了问题，请稍后再试。"

        debug_info["processing_time"] = time.time() - start_time
        return response, debug_info

    def _transition_state(self, new_state: DialogueState, debug_info: Dict):
        """状态转换"""
        old_state = self.current_state
        self.current_state = new_state
        debug_info["state_transitions"].append({"from": old_state.value, "to": new_state.value, "timestamp": time.time()})

    def _handle_clarification(self, intent_result: Dict, debug_info: Dict) -> str:
        """处理意图澄清"""
        user_input = intent_result.get("original_text", "")
        new_result, candidates = self.clarification_agent.clarify(user_input, self.context, self.dialogue_history)
        debug_info["clarification_candidates"] = candidates

        if new_result:
            clarification_type = new_result.get("clarification_type", "")

            if clarification_type == "auto_high_confidence":
                # 高置信度自澄清，直接执行
                debug_info["intent_result"] = new_result
                self._update_context(new_result, debug_info)
                return self._generate_response(new_result, debug_info)

            elif clarification_type == "user_confirmation_needed":
                # 中等置信度，需要用户确认
                suggested_text = new_result.get("suggested_interpretation", "")
                self.context["pending_confirmation"] = new_result
                debug_info["pending_confirmation"] = new_result
                return f"您是想说'{suggested_text}'吗？如果是，请回复'是'或'确认'；如果不是，请重新描述您的需求。"

        # 自澄清失败或低置信度，返回澄清问题
        return intent_result.get("clarification_question", "我不太确定您的意思，可以再说明一下吗？")

    def _check_user_confirmation(self, user_input: str, debug_info: Dict) -> bool:
        """检查用户是否在确认待执行的指令"""
        if "pending_confirmation" not in self.context:
            return False

        # 检查确认关键词
        confirmation_keywords = ["是", "对", "确认", "好的", "可以", "yes", "ok", "嗯"]
        rejection_keywords = ["不是", "不对", "不", "错了", "no", "重新"]

        user_input_lower = user_input.lower().strip()

        # 检查是否为确认
        if any(keyword in user_input_lower for keyword in confirmation_keywords):
            debug_info["confirmation_result"] = "confirmed"
            return True

        # 检查是否为拒绝
        if any(keyword in user_input_lower for keyword in rejection_keywords):
            # 清除待确认状态
            self.context.pop("pending_confirmation", None)
            debug_info["confirmation_result"] = "rejected"
            return False

        # 如果输入很短且不明确，可能是确认
        if len(user_input.strip()) <= 2 and user_input_lower in ["是", "对", "好", "嗯"]:
            debug_info["confirmation_result"] = "confirmed"
            return True

        return False

    def _generate_response(self, intent_result: Dict, debug_info: Dict) -> str:
        """生成系统响应"""
        try:
            messages = self._build_messages(intent_result)
            api_response = self.api_client.chat_completion(messages)
            debug_info["api_calls"].append(
                {
                    "success": api_response.success,
                    "content": api_response.content,
                    "error": api_response.error_message,
                    "response_time": api_response.response_time,
                    "request": {"messages": messages, "model": self.api_client.model_id},
                    "response": api_response.raw_response,
                }
            )

            if api_response.success:
                return api_response.content or "好的，我明白了。"
            else:
                self.logger.error(f"API调用失败: {api_response.error_message}")
                return "抱歉，我现在无法处理您的请求，请稍后再试。"
        except Exception as e:
            self.logger.error(f"生成响应时出错: {e}")
            return "系统出现问题，请稍后再试。"

    def _build_messages(self, intent_result: Dict) -> List[Dict]:
        """构建API请求消息"""
        messages = []

        # 系统提示
        system_prompt = "你是一个智能家居助手，负责理解用户指令并控制家电设备。请用简洁、友好的语言回复用户。"
        messages.append({"role": "system", "content": system_prompt})

        # 历史对话（最近3轮）
        for turn in self.dialogue_history[-3:]:
            messages.append({"role": "user", "content": turn.user_input})
            messages.append({"role": "assistant", "content": turn.system_response})

        # 添加当前用户输入（从intent_result中获取）
        current_input = intent_result.get("original_text", "")
        if current_input:
            messages.append({"role": "user", "content": current_input})

        return messages

    def _update_context(self, intent_result: Dict, debug_info: Dict):
        """更新对话上下文"""
        # 处理实体数据，将Entity对象列表转换为按类型分组的字典
        entities = intent_result.get("entities", [])
        grouped_entities = {"devices": [], "actions": [], "values": []}

        # 如果entities是字典列表（来自intent_result["entities"]），需要重新构建Entity对象
        for entity in entities:
            if isinstance(entity, dict):
                entity_type = entity.get("entity_type", "unknown")
                entity_value = entity.get("value", "")

                if entity_type == "device":
                    grouped_entities["devices"].append({"value": entity_value})
                elif entity_type == "action":
                    grouped_entities["actions"].append({"value": entity_value})
                elif entity_type in ["number", "value"]:
                    grouped_entities["values"].append({"value": entity_value})

        updates = {
            "last_intent": intent_result.get("intent"),
            "last_entities": grouped_entities,
            "last_possible_intents": intent_result.get("possible_intents", []),
            "turn_count": len(self.dialogue_history),
        }
        self.context.update(updates)
        debug_info["context_updates"] = updates

    def get_session_info(self) -> Dict:
        """获取会话信息"""
        return {
            "session_id": self.session_id,
            "current_state": self.current_state.value,
            "turn_count": len(self.dialogue_history),
            "context": self.context,
        }

    def end_session(self):
        """结束对话会话"""
        if self.session_id:
            # 保存会话到数据库
            self.memory_manager.save_session(self.session_id, self.dialogue_history)
            self.logger.info(f"Ended session: {self.session_id}")

        self.session_id = None
        self.current_state = DialogueState.IDLE
        self.dialogue_history.clear()
        self.context.clear()
