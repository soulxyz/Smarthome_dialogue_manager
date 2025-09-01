"""Clarification Agent

提供一个利用大模型(通过 SiliconFlowClient)对低置信度指令进行自我澄清的组件。
核心流程:
1. 通过提示工程, 让 LLM 生成若干候选澄清后的完整指令;
2. 使用现有 IntentRecognizer 对候选指令进行意图识别;
3. 选择置信度最高的一条, 若高于阈值则直接返回, 否则返回 None 以便后续由系统继续向用户澄清。

该模块专注于算法逻辑, 不涉及 UI 交互, UI 只负责在需要时向用户展示 ClarificationAgent 返回的 clarification_question 并获取确认结果。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .api_client import SiliconFlowClient
from .intent import IntentRecognizer

logger = logging.getLogger(__name__)


class ClarificationAgent:
    """基于 LLM 的澄清代理"""

    DEFAULT_PROMPT_TEMPLATE = (
        "你是一个智能家居助手, 需要帮助解析用户的模糊指令。\n"
        '当前用户输入: "{user_input}"\n'
        "历史对话上下文: {context_info}\n"
        "请基于上下文信息，生成不超过 {max_candidates} 条可能的完整指令改写。\n"
        "要求：\n"
        "1. 优先考虑与历史对话的关联性\n"
        "2. 如果上下文中有设备信息，优先补全相关设备操作\n"
        "3. 返回JSON列表格式，每条指令不超过20个中文字符\n"
        "4. 按照可能性从高到低排序"
    )

    def __init__(
        self,
        api_client: SiliconFlowClient,
        intent_recognizer: IntentRecognizer,
        confidence_threshold: float = 0.7,
        max_candidates: int = 3,
    ):
        self.api_client = api_client
        self.intent_recognizer = intent_recognizer
        self.confidence_threshold = confidence_threshold
        self.max_candidates = max_candidates

    # ---------------------------------------------------------------------
    # public api
    # ---------------------------------------------------------------------
    def clarify(
        self,
        user_input: str,
        context: Dict,
        history: List,
    ) -> Tuple[Optional[Dict], List[str]]:
        """尝试自澄清并返回新的 intent_result。

        Returns:
            (intent_result, candidates)
            intent_result 为空表示澄清未成功, 系统仍需向用户发问。
        """
        # 1. 让 LLM 生成候选澄清指令
        candidates = self._generate_candidates(user_input, context, history)
        if not candidates:
            return None, []

        best_result: Optional[Dict] = None
        best_conf = -1.0

        # 2. 对每个候选进行意图识别
        for cand in candidates:
            result = self.intent_recognizer.recognize(cand, context, history)

            # 如果IntentRecognizer没有识别出实体，使用兜底机制
            if not result.get("entities"):
                result["entities"] = self._simple_extract_entities(cand, context)

            conf = result.get("confidence", 0.0)
            if conf > best_conf:
                best_conf = conf
                best_result = result

        # 3. 分级判断置信度
        high_confidence_threshold = 0.8  # 高置信度阈值，直接执行
        medium_confidence_threshold = 0.5  # 中等置信度阈值，需要确认

        if best_result and best_conf >= high_confidence_threshold:
            # 高置信度：直接采用自澄清结果
            logger.info(
                "ClarificationAgent 高置信度自澄清: %s (conf=%.2f)",
                best_result.get("intent"),
                best_conf,
            )
            best_result["self_clarified"] = True
            best_result["clarification_type"] = "auto_high_confidence"
            best_result["clarified_text"] = best_result.get("original_text", "")
            return best_result, candidates

        elif best_result and best_conf >= medium_confidence_threshold:
            # 中等置信度：需要用户确认
            logger.info(
                "ClarificationAgent 中等置信度，需要用户确认: %s (conf=%.2f)",
                best_result.get("intent"),
                best_conf,
            )
            best_result["self_clarified"] = False
            best_result["clarification_type"] = "user_confirmation_needed"
            best_result["clarified_text"] = best_result.get("original_text", "")
            best_result["suggested_interpretation"] = best_result.get("original_text", "")
            return best_result, candidates

        # 低置信度：返回None，需要系统向用户澄清
        return None, candidates

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _generate_candidates(self, user_input: str, context: Dict = None, history: List = None) -> List[str]:
        """调用 LLM 生成候选澄清指令列表"""
        # 构建上下文信息
        context_info = self._build_context_info(context, history)

        prompt = self.DEFAULT_PROMPT_TEMPLATE.format(
            user_input=user_input, max_candidates=self.max_candidates, context_info=context_info
        )
        messages = [
            {"role": "system", "content": "你是一个严谨的 JSON 生成器。"},
            {"role": "user", "content": prompt},
        ]

        try:
            # 为澄清任务使用较短的超时时间，避免用户等待过久
            original_timeout = self.api_client.timeout
            self.api_client.timeout = 10  # 临时设置为10秒超时

            resp = self.api_client.chat_completion(messages, max_tokens=500, temperature=0.3)

            # 恢复原始超时设置
            self.api_client.timeout = original_timeout

            if not resp.success:
                logger.warning("Clarification LLM 失败: %s", resp.error_message)
                # 如果是服务器繁忙错误或超时错误，返回基于规则的候选
                if any(keyword in resp.error_message.lower() for keyword in ["503", "busy", "timeout", "超时"]):
                    logger.info("检测到API超时或服务繁忙，使用降级策略")
                    return self._generate_fallback_candidates(user_input, context, history)
                return []

            # 尝试解析 JSON 列表
            text = resp.content.strip()
            if not text:
                logger.warning("LLM 返回空响应，使用降级策略")
                return self._generate_fallback_candidates(user_input, context, history)

            # 尝试多种JSON解析策略
            candidates = self._parse_candidates_response(text)
            if candidates:
                return candidates
            else:
                logger.warning("JSON解析失败，使用降级策略")
                return self._generate_fallback_candidates(user_input, context, history)

        except Exception as exc:
            # 确保恢复原始超时设置
            if "original_timeout" in locals():
                self.api_client.timeout = original_timeout
            logger.error("解析 Clarification LLM 输出失败: %s", exc)
            return self._generate_fallback_candidates(user_input, context, history)

    def _parse_candidates_response(self, text: str) -> List[str]:
        """尝试多种方式解析LLM响应"""
        try:
            # 方法1: 直接解析JSON
            candidates = json.loads(text)
            if isinstance(candidates, list):
                return [str(c).strip() for c in candidates if str(c).strip()]
        except json.JSONDecodeError:
            pass

        try:
            # 方法2: 提取JSON数组部分
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                candidates = json.loads(json_match.group())
                if isinstance(candidates, list):
                    return [str(c).strip() for c in candidates if str(c).strip()]
        except (json.JSONDecodeError, AttributeError):
            pass

        try:
            # 方法3: 按行分割并清理
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            candidates = []
            for line in lines:
                # 移除引号、序号等
                clean_line = re.sub(r'^[\d\.\-\*\s]*["\'\']?', "", line)
                clean_line = re.sub(r'["\'\']?[,\s]*$', "", clean_line)
                if clean_line and len(clean_line) > 2:
                    candidates.append(clean_line)
            if candidates:
                return candidates[: self.max_candidates]
        except Exception:
            pass

        return []

    def _simple_extract_entities(self, text: str, context: Dict) -> List[Dict]:
        """
简易实体抽取兜底机制.

Args:
    text (str): 输入文本.
    context (Dict): 上下文信息.

Returns:
    List[Dict]: 提取的实体列表.
"""
        entities = []

        # 从上下文中获取最近的设备和动作
        last_entities = context.get("last_entities", {})
        recent_devices = {entity["value"] for entity in last_entities.get("devices", [])}
        recent_actions = {entity["value"] for entity in last_entities.get("actions", [])}

        # 设备词典（扩展版）
        device_dict = {
            "空调": ["空调", "冷气", "暖气", "制冷", "制热"],
            "电视": ["电视", "电视机", "TV", "显示器"],
            "灯": ["灯", "台灯", "吊灯", "壁灯", "射灯", "客厅灯", "卧室灯", "厨房灯"],
            "风扇": ["风扇", "吊扇", "落地扇", "台扇"],
            "热水器": ["热水器", "热水", "水温"],
            "洗衣机": ["洗衣机", "洗衣", "脱水", "甩干"],
        }

        # 动作词典
        action_dict = {
            "开启": ["打开", "开启", "启动", "开", "打开"],
            "关闭": ["关闭", "关掉", "关", "停止", "关掉"],
            "调节": ["调节", "设置", "调到", "调整"],
            "增加": ["增加", "提高", "调高", "加大"],
            "减少": ["减少", "降低", "调低", "减小"],
        }

        # 优先匹配上下文中的设备
        for device in recent_devices:
            if device in text:
                entities.append({"entity_type": "device", "value": device, "name": device, "confidence": 0.9})

        # 优先匹配上下文中的动作
        for action in recent_actions:
            if action in text:
                entities.append({"entity_type": "action", "value": action, "name": action, "confidence": 0.9})

        # 如果上下文匹配不到，使用词典匹配
        if not any(e["entity_type"] == "device" for e in entities):
            for device_type, keywords in device_dict.items():
                for keyword in keywords:
                    if keyword in text:
                        entities.append({"entity_type": "device", "value": device_type, "name": keyword, "confidence": 0.8})
                        break

        if not any(e["entity_type"] == "action" for e in entities):
            for action_type, keywords in action_dict.items():
                for keyword in keywords:
                    if keyword in text:
                        entities.append({"entity_type": "action", "value": action_type, "name": keyword, "confidence": 0.8})
                        break

        # 抽取数值实体（温度、档位等）
        number_patterns = [r"(\d+)\s*度", r"(\d+)\s*档", r"(\d+)\s*级", r"(\d+)\s*%", r"(\d+)\s*点", r"(\d+)"]

        for pattern in number_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({"entity_type": "number", "value": match.group(1), "name": match.group(0), "confidence": 0.9})
                break  # 只取第一个数字

        # 处理指代词（如"跟往常一样"、"老样子"等）
        reference_patterns = [
            r"跟?往常一样",
            r"老样子",
            r"跟?上次一样",
            r"跟?之前一样",
            r"还是?那样",
            r"照旧",
            r"跟?平时一样",
            r"跟?以前一样",
        ]

        for pattern in reference_patterns:
            if re.search(pattern, text):
                # 从上下文中获取最近的完整操作信息
                recent_devices = last_entities.get("devices", [])
                recent_actions = last_entities.get("actions", [])
                recent_values = last_entities.get("values", [])

                # 优先补全缺失的设备信息
                if recent_devices and not any(e["entity_type"] == "device" for e in entities):
                    last_device = recent_devices[-1]["value"]
                    entities.append({"entity_type": "device", "value": last_device, "name": "往常设备", "confidence": 0.85})

                # 补全动作信息
                if recent_actions and not any(e["entity_type"] == "action" for e in entities):
                    last_action = recent_actions[-1]["value"]
                    entities.append({"entity_type": "action", "value": last_action, "name": "往常操作", "confidence": 0.85})

                # 补全数值信息
                if recent_values and not any(e["entity_type"] == "number" for e in entities):
                    last_value = recent_values[-1]["value"]
                    entities.append({"entity_type": "number", "value": last_value, "name": "往常设置", "confidence": 0.85})
                break

        return entities

    def _build_context_info(self, context: Dict = None, history: List = None) -> str:
        """
构建上下文信息字符串.

Args:
    context (Dict, optional): 上下文字典.
    history (List, optional): 历史对话列表.

Returns:
    str: 构建的上下文信息字符串.
"""
        context_parts = []

        # 从context中提取关键信息
        if context:
            # 提取最近的设备实体
            last_entities = context.get("last_entities", {})
            if last_entities:
                devices = [entity["value"] for entity in last_entities.get("devices", [])]
                if devices:
                    context_parts.append(f"最近操作的设备: {', '.join(devices)}")

                actions = [entity["value"] for entity in last_entities.get("actions", [])]
                if actions:
                    context_parts.append(f"最近的操作: {', '.join(actions)}")

                values = [entity["value"] for entity in last_entities.get("values", [])]
                if values:
                    context_parts.append(f"最近的参数: {', '.join(values)}")

            # 提取最近的意图
            last_intent = context.get("last_intent")
            if last_intent:
                context_parts.append(f"上一轮意图: {last_intent}")

        # 从history中提取最近的对话
        if history and len(history) > 0:
            recent_dialogues = []
            # 只取最近3轮对话
            for dialogue in history[-3:]:
                if isinstance(dialogue, dict):
                    user_text = dialogue.get("user_input", "")
                    system_text = dialogue.get("system_response", "")
                    if user_text:
                        recent_dialogues.append(f"用户: {user_text}")
                    if system_text and len(system_text) < 50:  # 避免响应过长
                        recent_dialogues.append(f"系统: {system_text}")

            if recent_dialogues:
                context_parts.append("最近对话:\n" + "\n".join(recent_dialogues))

        return "\n".join(context_parts) if context_parts else "无相关上下文"

    def _generate_fallback_candidates(self, user_input: str, context: Dict = None, history: List = None) -> List[str]:
        """
基于规则和上下文的降级候选生成.

Args:
    user_input (str): 用户输入.
    context (Dict, optional): 上下文字典.
    history (List, optional): 历史对话列表.

Returns:
    List[str]: 候选列表.
"""
        candidates = []

        # 从上下文中获取最近的设备信息
        recent_devices = []
        if context and context.get("last_entities", {}).get("devices"):
            recent_devices = [entity["value"] for entity in context["last_entities"]["devices"]]

        # 基于关键词的模板扩展
        templates = [
            f"我想{user_input}",
            f"请帮我{user_input}",
            f"能否{user_input}",
            f"我需要{user_input}",
        ]

        # 设备控制相关的扩展
        if any(word in user_input for word in ["度", "温", "热", "冷"]):
            # 优先使用上下文中的设备
            if recent_devices:
                for device in recent_devices:
                    candidates.extend(
                        [f"调节{device}温度到{user_input}", f"设置{device}{user_input}", f"{device}温度{user_input}"]
                    )
            else:
                candidates.extend([f"调节空调温度到{user_input}", f"设置温度{user_input}", f"空调温度{user_input}"])

        if any(word in user_input for word in ["开", "关", "启动", "停止"]):
            # 优先使用上下文中的设备
            if recent_devices:
                for device in recent_devices:
                    candidates.append(f"{user_input}{device}")
            else:
                candidates.extend([f"{user_input}空调", f"{user_input}电视", f"{user_input}灯光"])

        # 如果用户输入很短且上下文中有设备，尝试组合
        if len(user_input) <= 3 and recent_devices:
            for device in recent_devices:
                candidates.extend([f"{user_input}{device}", f"{device}{user_input}", f"设置{device}{user_input}"])

        # 添加通用模板
        for template in templates:
            if template not in candidates:
                candidates.append(template)

        return candidates[: self.max_candidates]
