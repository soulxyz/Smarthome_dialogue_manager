"""意图识别模块

负责用户输入的意图分类、实体抽取和置信度评估。
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class IntentType(Enum):
    """意图类型枚举"""

    DEVICE_CONTROL = "device_control"  # 设备控制
    QUERY_STATUS = "query_status"  # 查询状态
    SCENE_CONTROL = "scene_control"  # 场景控制
    SCHEDULE_TASK = "schedule_task"  # 定时任务
    SYSTEM_CONFIG = "system_config"  # 系统配置
    GREETING = "greeting"  # 问候
    QUERY_WEATHER = "query_weather"  # 天气查询
    UNKNOWN = "unknown"  # 未知意图


@dataclass
class Entity:
    """实体数据结构"""

    name: str  # 实体名称
    value: str  # 实体值
    entity_type: str  # 实体类型
    confidence: float  # 置信度
    start_pos: int = -1  # 开始位置
    end_pos: int = -1  # 结束位置


@dataclass
class IntentResult:
    """意图识别结果"""

    intent: str  # 识别的意图
    confidence: float  # 置信度
    entities: List[Entity]  # 抽取的实体
    possible_intents: List[str]  # 可能的意图列表
    need_clarification: bool  # 是否需要澄清
    clarification_question: str = ""  # 澄清问题


class IntentRecognizer:
    """意图识别器"""

    def __init__(self, confidence_threshold: float = 0.7):
        """初始化意图识别器

        Args:
            confidence_threshold: 置信度阈值，低于此值需要澄清
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

        # 初始化意图模式和实体模式
        self._init_patterns()
        # 动态模式：会话级新增正则 {intent: [patterns]}
        self.session_patterns: Dict[str, List[str]] = {}

    def _init_patterns(self):
        """初始化识别模式"""
        # 设备控制相关关键词
        self.device_control_patterns = [
            r"(打开|开启|启动|开).*?(灯|空调|电视|风扇|热水器|洗衣机)",
            r"(关闭|关掉|关|停止).*?(灯|空调|电视|风扇|热水器|洗衣机)",
            r"(调节|设置|调到).*?(温度|亮度|音量|风速)",
            r"(增加|提高|调高).*?(温度|亮度|音量|风速)",
            r"(减少|降低|调低).*?(温度|亮度|音量|风速)",
        ]

        # 查询状态相关关键词
        self.query_patterns = [
            r"(查看|查询|看看|检查).*?(状态|情况|温度|湿度)",
            r".*?(怎么样|如何|状态|情况)",
            r"(现在|当前).*?(温度|湿度|状态)",
        ]

        # 场景控制相关关键词
        self.scene_patterns = [
            r"(启动|开启|切换到).*?(睡眠|观影|聚会|离家|回家).*?模式",
            r"(睡眠|观影|聚会|离家|回家).*?模式",
            r"我要(睡觉|看电影|开派对|出门|回家)",
        ]

        # 定时任务相关关键词
        self.schedule_patterns = [
            r"(定时|预约|设置).*?(开启|关闭|启动).*?(\d+)(点|小时|分钟)",
            r"(\d+)(点|小时|分钟).*?(后|之后).*?(开启|关闭|启动)",
            r"每天.*?(\d+)点.*?(开启|关闭|启动)",
        ]

        # 问候相关关键词
        self.greeting_patterns = [r"^(你好|您好|hi|hello|嗨)", r"(早上好|下午好|晚上好|晚安)", r"(谢谢|感谢|再见|拜拜)"]
        
        # 天气查询相关关键词
        self.weather_patterns = [
            r".*?(天气|气温|温度|下雨|晴天|阴天|雨天).*?",
            r"(今天|明天|后天|昨天).*?(天气|气温|温度)",
            r".*?(怎么样|如何).*?(天气|气温)",
            r"(那|这).*?(明天|今天|后天|昨天).*?(呢|怎么样|如何)",
            r"(明天|今天|后天|昨天).*?(呢|怎么样|如何)"
        ]

        # 设备实体模式
        self.device_entities = {
            "灯": ["灯", "台灯", "吊灯", "壁灯", "射灯", "客厅灯", "卧室灯", "厨房灯"],
            "空调": ["空调", "冷气", "暖气", "制冷", "制热"],
            "电视": ["电视", "电视机", "TV", "显示器"],
            "风扇": ["风扇", "吊扇", "落地扇", "台扇"],
            "热水器": ["热水器", "热水", "水温"],
            "洗衣机": ["洗衣机", "洗衣", "脱水", "甩干"],
        }

        # 动作实体模式
        self.action_entities = {
            "开启": ["打开", "开启", "启动", "开"],
            "关闭": ["关闭", "关掉", "关", "停止"],
            "调节": ["调节", "设置", "调到", "调整"],
            "增加": ["增加", "提高", "调高", "加大"],
            "减少": ["减少", "降低", "调低", "减小"],
        }

    def recognize(self, user_input: str, context: Dict, history: List) -> Dict:
        """识别用户输入的意图

        Args:
            user_input: 用户输入文本
            context: 对话上下文
            history: 对话历史

        Returns:
            Dict: 意图识别结果
        """
        try:
            # 预处理输入
            processed_input = self._preprocess_input(user_input)

            # 意图分类
            intent_scores = self._classify_intent(processed_input)

            # 实体抽取（结合历史记录）
            entities = self._extract_entities_with_history(processed_input, history)

            # 上下文增强
            intent_scores = self._enhance_with_context(user_input, intent_scores, context, history)

            # 选择最佳意图
            best_intent, confidence = self._select_best_intent(intent_scores)

            # 判断是否需要澄清
            need_clarification = confidence < self.confidence_threshold
            possible_intents = [
                intent for intent, score in sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            ]

            # 生成澄清问题
            clarification_question = ""
            if need_clarification:
                clarification_question = self._generate_clarification_question(possible_intents, entities)

            result = {
                "intent": best_intent,
                "confidence": confidence,
                "entities": [entity.__dict__ for entity in entities],
                "possible_intents": possible_intents,
                "need_clarification": need_clarification,
                "clarification_question": clarification_question,
                "original_text": user_input,
            }

            self.logger.info(f"Intent recognition result: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error in intent recognition: {e}")
            return {
                "intent": IntentType.UNKNOWN.value,
                "confidence": 0.0,
                "entities": [],
                "possible_intents": [IntentType.UNKNOWN.value],
                "need_clarification": True,
                "clarification_question": "抱歉，我没有理解您的意思，请您再说一遍。",
                "original_text": user_input,
            }

    def _preprocess_input(self, user_input: str) -> str:
        """预处理用户输入"""
        # 转换为小写，去除多余空格
        processed = user_input.strip().lower()
        # 去除标点符号（保留中文标点）
        processed = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", processed)
        return processed

    def set_session_patterns(self, patterns: Dict[str, List[str]]):
        """注入会话级自学习正则模式.

        Args:
            patterns: 会话模式字典
        """
        self.session_patterns = patterns or {}

    def _classify_intent(self, user_input: str) -> Dict[str, float]:
        """意图分类.

        Args:
            user_input: 用户输入

        Returns:
            Dict[str, float]: 意图分数
        """
        scores = {intent.value: 0.0 for intent in IntentType}

        # 设备控制意图
        for pattern in self.device_control_patterns:
            if re.search(pattern, user_input):
                scores[IntentType.DEVICE_CONTROL.value] += 0.8

        # 查询状态意图
        for pattern in self.query_patterns:
            if re.search(pattern, user_input):
                scores[IntentType.QUERY_STATUS.value] += 0.7

        # 场景控制意图
        for pattern in self.scene_patterns:
            if re.search(pattern, user_input):
                scores[IntentType.SCENE_CONTROL.value] += 0.8

        # 定时任务意图
        for pattern in self.schedule_patterns:
            if re.search(pattern, user_input):
                scores[IntentType.SCHEDULE_TASK.value] += 0.7

        # 问候意图
        for pattern in self.greeting_patterns:
            if re.search(pattern, user_input):
                scores[IntentType.GREETING.value] += 0.9
                
        # 天气查询意图
        for pattern in self.weather_patterns:
            if re.search(pattern, user_input):
                scores[IntentType.QUERY_WEATHER.value] += 0.8

        # 使用会话级动态模式
        for intent_key, pattern_list in self.session_patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, user_input):
                    # 动态模式匹配给予更高权重
                    scores[intent_key] += 1.0
        return scores

    def _extract_entities(self, user_input: str) -> List[Entity]:
        """实体抽取, 避免重叠实体"""
        entities = []
        found_spans = []

        # 1. 组合设备和动作实体关键词
        all_keywords = {}
        for device_type, keywords in self.device_entities.items():
            for keyword in set(keywords):
                all_keywords[keyword] = ("device", device_type, 0.9)

        for action_type, keywords in self.action_entities.items():
            for keyword in set(keywords):
                all_keywords[keyword] = ("action", action_type, 0.8)

        # 按长度降序排序关键词，以优先匹配长实体
        sorted_keywords = sorted(all_keywords.keys(), key=len, reverse=True)

        # 2. 抽取设备和动作实体
        for keyword in sorted_keywords:
            entity_type, value, confidence = all_keywords[keyword]
            for match in re.finditer(re.escape(keyword), user_input):
                start, end = match.span()

                # 检查此范围是否与已找到的实体范围重叠
                is_overlapping = False
                for found_start, found_end in found_spans:
                    if start < found_end and end > found_start:
                        is_overlapping = True
                        break

                if not is_overlapping:
                    entity = Entity(
                        name=keyword,
                        value=value,
                        entity_type=entity_type,
                        confidence=confidence,
                        start_pos=start,
                        end_pos=end,
                    )
                    entities.append(entity)
                    found_spans.append((start, end))

        # 3. 抽取数值实体
        number_pattern = r"(\d+)\s*(度|点|小时|分钟|%|档)"
        for match in re.finditer(number_pattern, user_input):
            start, end = match.span()
            is_overlapping = False
            for found_start, found_end in found_spans:
                if start < found_end and end > found_start:
                    is_overlapping = True
                    break

            if not is_overlapping:
                entity = Entity(
                    name=match.group(0),
                    value=match.group(1),
                    entity_type="number",
                    confidence=0.9,
                    start_pos=start,
                    end_pos=end,
                )
                entities.append(entity)
                found_spans.append((start, end))

        # 4. 抽取地点实体
        location_keywords = ["北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "成都", "重庆", "西安"]
        for location in sorted(location_keywords, key=len, reverse=True):
            for match in re.finditer(re.escape(location), user_input):
                start, end = match.span()
                is_overlapping = False
                for found_start, found_end in found_spans:
                    if start < found_end and end > found_start:
                        is_overlapping = True
                        break

                if not is_overlapping:
                    entity = Entity(
                        name=location,
                        value=location,
                        entity_type="location",
                        confidence=0.9,
                        start_pos=start,
                        end_pos=end,
                    )
                    entities.append(entity)
                    found_spans.append((start, end))

        return entities

    def _enhance_with_context(
        self, user_input: str, intent_scores: Dict[str, float], context: Dict, history: List
    ) -> Dict[str, float]:
        """使用上下文增强意图识别"""
        # 如果上一轮是澄清，提高相关意图的分数
        if context.get("last_intent") == "clarification":
            last_possible_intents = context.get("last_possible_intents", [])
            for intent in last_possible_intents:
                if intent in intent_scores:
                    intent_scores[intent] += 0.2

        # 根据历史对话调整分数
        if history:
            recent_intents = []
            for turn in history[-3:]:
                if hasattr(turn, 'intent') and turn.intent:
                    recent_intents.append(turn.intent)
                elif isinstance(turn, dict) and turn.get('intent'):
                    recent_intents.append(turn['intent'])
            for intent in recent_intents:
                if intent in intent_scores:
                    intent_scores[intent] += 0.1

        return intent_scores
        
    def _extract_entities_with_history(self, user_input: str, history: List) -> List[Entity]:
        """结合历史记录提取实体"""
        entities = self._extract_entities(user_input)
        
        # 从历史记录中提取相关实体
        if history:
            for turn in history[-2:]:  # 检查最近2轮对话
                if isinstance(turn, dict) and turn.get('entities'):
                    for hist_entity in turn['entities']:
                        if isinstance(hist_entity, dict):
                            # 如果当前输入中没有地点实体，但历史中有，则继承
                            if (hist_entity.get('type') == 'location' and 
                                not any(e.entity_type == 'location' for e in entities)):
                                entity = Entity(
                                    name=hist_entity['value'],
                                    value=hist_entity['value'],
                                    entity_type='location',
                                    confidence=0.7,  # 历史实体置信度稍低
                                    start_pos=0,
                                    end_pos=0,
                                )
                                entities.append(entity)
        
        return entities

    def _select_best_intent(self, intent_scores: Dict[str, float]) -> Tuple[str, float]:
        """选择最佳意图"""
        if not intent_scores or max(intent_scores.values()) == 0:
            return IntentType.UNKNOWN.value, 0.0

        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(intent_scores[best_intent], 1.0)  # 确保置信度不超过1.0

        return best_intent, confidence

    def _generate_clarification_question(self, possible_intents: List[str], entities: List[Entity]) -> str:
        """生成澄清问题"""
        if not possible_intents:
            return "抱歉，我没有理解您的意思，请您再详细说明一下。"

        # 根据可能的意图生成澄清问题
        intent_questions = {
            IntentType.DEVICE_CONTROL.value: "您是想要控制某个设备吗？",
            IntentType.QUERY_STATUS.value: "您是想要查询设备状态吗？",
            IntentType.SCENE_CONTROL.value: "您是想要切换场景模式吗？",
            IntentType.SCHEDULE_TASK.value: "您是想要设置定时任务吗？",
        }

        top_intent = possible_intents[0]
        if top_intent in intent_questions:
            return intent_questions[top_intent]

        return "请问您具体想要做什么呢？"

    def update_patterns(self, new_patterns: Dict):
        """更新识别模式（用于在线学习）.

        Args:
            new_patterns: 新模式字典
        """
        # TODO: 实现模式更新逻辑
        pass

    def get_statistics(self) -> Dict:
        """获取识别统计信息.

        Returns:
            Dict: 统计信息
        """
        # TODO: 实现统计信息收集
        return {"total_recognitions": 0, "clarification_rate": 0.0, "intent_distribution": {}}
