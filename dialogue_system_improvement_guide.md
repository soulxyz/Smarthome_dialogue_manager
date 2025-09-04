# 对话系统多轮对话能力改进指南（优先级排序版）

## 📋 改进概述

本文档基于对现有对话系统的全面评估，提供详细的改进方案和实施指导。当前系统在焦点实体管理和代词消解方面已有基础实现，但在省略消解、LLM回退机制、错误处理等方面仍需完善。

### 🎯 改进目标
- 完善省略主语的指令处理能力
- 实现智能的LLM回退机制
- 增强系统的错误处理和恢复能力
- 提升测试覆盖率和代码质量
- 优化系统架构的可维护性

### 🏗️ 实施策略
**搭积木式分阶段推进**：按优先级从高到低，逐步完成各项改进，确保每个阶段都有明确的交付物和验收标准。避免一次性大规模重构带来的风险。

### ⚡ 优先级分级
- **P0 核心可用性**：不依赖LLM即可稳定处理省略指令和焦点切换
- **P1 智能回退**：在规则失效时启用LLM最小回退机制
- **P2 错误恢复**：完善异常处理和幂等性控制
- **P3 可观测性**：结构化日志和性能监控
- **P4 性能优化**：并发处理和成本控制
- **P5 扩展功能**：多语言支持和上下文持久化

## 🔧 核心功能改进

### P0 核心可用性改进（第一优先级）

#### 1. 省略消解功能完善 🎯

**问题描述**：当前系统无法有效处理省略主语的指令（如"关掉"、"调亮一点"）。

**解决方案**：在 `dialogue_manager/intent.py` 中增强实体抽取逻辑。

**实施时间**：1周内完成
**风险等级**：低

```python
# 在 IntentRecognizer 类中添加以下方法
def _extract_entities_with_context(self, user_input: str, intent: str, context: Dict) -> List[Entity]:
    """结合上下文进行实体抽取，支持省略消解"""
    entities = self._extract_entities(user_input, intent)
    
    # 检查是否缺少设备实体但有动作实体
    has_device = any(e.entity_type == "device" for e in entities)
    has_action = any(e.entity_type == "action" for e in entities)
    
    if not has_device and has_action and "current_focus" in context:
        # 从焦点实体补充设备实体
        focus_entity = context["current_focus"].get("value")
        if focus_entity:
            # 创建焦点设备实体
            focus_device_entity = Entity(
                name=focus_entity,
                value=focus_entity,
                entity_type="device",
                confidence=0.8,  # 来自上下文的置信度稍低
                start_pos=-1,
                end_pos=-1
            )
            entities.append(focus_device_entity)
            self.logger.info(f"Added focus entity '{focus_entity}' for omitted subject")
    
    return entities
```

**修改位置**：
1. 将 `recognize` 方法中的 `_extract_entities_with_history` 调用替换为 `_extract_entities_with_context`
2. 更新相关的实体抽取逻辑

**验收标准**：
- 省略主语指令（如"关掉""调亮一点"）在存在有效焦点时成功执行
- 通过单元测试覆盖主要边界条件
- 误用焦点的情况下提供澄清提示

#### 2. 智能焦点切换优化 🎯

**问题描述**：当前焦点切换逻辑过于简单，缺乏语义相关性判断。

**解决方案**：增强焦点管理的智能性。

**实施时间**：0.5周
**风险等级**：中等

```python
# 在 DialogueEngine 类中添加
def _should_switch_focus(self, new_entity: str, current_focus: Dict, intent: str) -> bool:
    """判断是否应该切换焦点实体"""
    if not current_focus:
        return True
    
    current_entity = current_focus.get("value")
    if not current_entity:
        return True
    
    # 如果是相同实体，不切换
    if new_entity == current_entity:
        return False
    
    # 如果是查询意图，通常不切换焦点
    if intent == "query_status":
        return False
    
    # 如果新实体是明确的设备控制指令，切换焦点
    if intent == "device_control":
        return True
    
    # 其他情况保持当前焦点
    return False
```

**验收标准**：
- 焦点切换符合策略，query不轻易改变焦点
- 相同实体不重复切换
- 通过测试用例验证切换逻辑

### P1 智能回退机制（第二优先级）

#### 3. LLM回退机制实现 🤖

**问题描述**：当正则表达式无法识别复杂或模糊意图时，系统缺乏智能回退机制。

**解决方案**：在 `dialogue_manager/engine.py` 中实现LLM回退功能。

**实施时间**：1周
**风险等级**：中等
**成本控制**：需要设置预算阈值和超时机制

```python
# 在 DialogueEngine 类中添加以下方法
def _get_intent_from_llm(self, user_input: str, context: Dict, debug_info: Dict) -> Dict:
    """使用LLM进行意图识别的回退机制"""
    try:
        # 构建LLM提示
        prompt = self._build_llm_intent_prompt(user_input, context)
        
        # 调用LLM API
        api_response = self.api_client.chat_completion([{"role": "user", "content": prompt}])
        
        # 记录API调用
        debug_info["api_calls"].append({
            "success": api_response.success,
            "content": api_response.content,
            "error": api_response.error_message,
            "response_time": api_response.response_time,
            "request": {"prompt": prompt, "model": self.api_client.model_id, "purpose": "intent_fallback"},
            "response": api_response.raw_response,
        })
        
        if api_response.success and api_response.content:
            # 解析LLM返回的JSON结果
            return self._parse_llm_intent_response(api_response.content)
        else:
            self.logger.error(f"LLM intent fallback failed: {api_response.error_message}")
            return self._create_unknown_intent_result(user_input)
            
    except Exception as e:
        self.logger.error(f"LLM intent fallback error: {e}")
        return self._create_unknown_intent_result(user_input)

def _build_llm_intent_prompt(self, user_input: str, context: Dict) -> str:
    """构建LLM意图识别提示"""
    prompt = f"""你是一个智能家居控制助手，请分析用户输入并返回意图识别结果。

用户输入：{user_input}

上下文信息：
"""
    
    # 添加焦点实体信息
    if "current_focus" in context:
        focus = context["current_focus"]
        prompt += f"- 当前焦点设备：{focus.get('value', 'unknown')}\n"
    
    # 添加最近的对话历史
    if self.dialogue_history:
        prompt += "- 最近对话：\n"
        for turn in self.dialogue_history[-2:]:
            prompt += f"  用户：{turn.user_input}\n"
            prompt += f"  系统：{turn.system_response}\n"
    
    prompt += """\n请返回JSON格式的意图识别结果，包含以下字段：
{
  "intent": "意图类型(device_control/query_status/scene_control/greeting/query_weather/unknown)",
  "confidence": 0.0-1.0,
  "entities": [
    {
      "name": "实体名称",
      "value": "实体值",
      "entity_type": "实体类型(device/action/number/location)",
      "confidence": 0.0-1.0
    }
  ],
  "possible_intents": ["可能的意图列表"],
  "need_clarification": false,
  "clarification_question": ""
}

只返回JSON，不要其他内容。"""
    
    return prompt

def _parse_llm_intent_response(self, response_content: str) -> Dict:
    """解析LLM返回的意图识别结果"""
    try:
        import json
        result = json.loads(response_content.strip())
        
        # 验证必要字段
        required_fields = ["intent", "confidence", "entities"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # 添加原始文本
        result["original_text"] = self.dialogue_history[-1].user_input if self.dialogue_history else ""
        
        return result
        
    except (json.JSONDecodeError, ValueError) as e:
        self.logger.error(f"Failed to parse LLM intent response: {e}")
        return self._create_unknown_intent_result(response_content)

def _create_unknown_intent_result(self, user_input: str) -> Dict:
    """创建未知意图的默认结果"""
    return {
        "intent": "unknown",
        "confidence": 0.1,
        "entities": [],
        "possible_intents": ["unknown"],
        "need_clarification": True,
        "clarification_question": "抱歉，我没有理解您的意思，请您换个方式描述。",
        "original_text": user_input
    }
```

**集成到主流程**：在 `process_input` 方法中添加LLM回退逻辑：

```python
# 在 process_input 方法中，意图识别后添加：
if intent_result["confidence"] < 0.4:  # 低置信度阈值
    self.logger.info(f"Low confidence ({intent_result['confidence']}), trying LLM fallback")
    llm_result = self._get_intent_from_llm(user_input, self.context, debug_info)
    if llm_result["confidence"] > intent_result["confidence"]:
        intent_result = llm_result
        debug_info["llm_fallback_used"] = True
```

**验收标准**：
- LLM回退命中时，复杂/模糊指令的识别准确率提升5%+
- JSON解析失败时安全回退到unknown+澄清提问
- 平均响应时间控制在可接受范围内（P95 ≤ 2.5s）
- 回退触发率 < 20%，其中70%为有效纠正

**配置参数建议**：
```python
# 在引擎配置中添加
min_confidence_for_llm: 0.4
llm_fallback_enabled: true  # 灰度开启
llm_timeout: 3  # 秒
per_turn_max_tokens: 500
per_session_budget: 1000  # 根据业务预算设定
```

### P2 错误处理与恢复（第三优先级）

#### 4. 异常处理和幂等性控制 🛡️

**问题描述**：系统缺乏完善的错误处理机制，失败路径行为不明确，可能产生重复副作用。

**解决方案**：建立分层异常处理体系和幂等性控制机制。

**实施时间**：0.5-1周
**风险等级**：低

**自定义异常类**：
```python
# 创建 dialogue_manager/exceptions.py
class DialogueException(Exception):
    """对话系统基础异常"""
    pass

class APIException(DialogueException):
    """API调用异常"""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code

class ParsingException(DialogueException):
    """解析异常"""
    pass

class DeviceException(DialogueException):
    """设备操作异常"""
    def __init__(self, message: str, device_id: str = None):
        super().__init__(message)
        self.device_id = device_id
```

**验收标准**：
- 人为注入网络故障、API 429/5xx、JSON解析失败时，系统不崩溃且返回合理提示
- 重放同一轮设备操作不重复执行
- 错误率（非用户输入导致）≤ 1%

### P3 可观测性增强（第四优先级）

#### 5. 结构化日志与性能监控 📊

**问题描述**：缺乏系统化的日志记录和性能监控，难以定位问题和量化收益。

**解决方案**：建立结构化日志体系和性能指标收集机制。

**实施时间**：0.5周
**风险等级**：低

**关键日志字段**：
- request_id、session_id、user_input、intent、confidence
- entities、current_focus、llm_fallback_used、api_usage
- latency_ms、error_code、error_stage

**性能指标**：
- 意图识别耗时、LLM回退耗时、总响应时延
- token用量、错误率、焦点切换频率

**验收标准**：
- 单次会话的关键决策链可追溯
- 延迟与错误分布可量化
- PII数据已脱敏处理

### P4-P5 扩展功能（后续优先级）

#### 6. 性能优化与多语言支持

**包含内容**：
- 并发处理与线程安全
- 缓存与限流策略
- 词法归一化（口语别称映射）
- 上下文长期持久化
- 焦点过期策略

**实施时间**：1-2周
**优先级**：根据业务需求调整

## 🧪 测试策略与用例

### 1. 边界条件测试

创建 `tests/test_edge_cases.py`：

```python
import unittest
from dialogue_manager.engine import DialogueEngine, EngineConfig

class TestEdgeCases(unittest.TestCase):
    
    def setUp(self):
        config = EngineConfig(enable_device_manager=True)
        self.engine = DialogueEngine(api_key="test_key", config=config)
        self.engine.start_session("test_user")
    
    def test_empty_input(self):
        """测试空输入处理"""
        response, debug_info = self.engine.process_input("")
        self.assertIn("请您说点什么", response)
    
    def test_very_long_input(self):
        """测试超长输入处理"""
        long_input = "打开" * 1000 + "灯"
        response, debug_info = self.engine.process_input(long_input)
        self.assertIsNotNone(response)
    
    def test_special_characters(self):
        """测试特殊字符输入"""
        special_input = "打开@#$%^&*()灯！！！"
        response, debug_info = self.engine.process_input(special_input)
        self.assertIsNotNone(response)
    
    def test_mixed_language(self):
        """测试中英文混合输入"""
        mixed_input = "turn on 客厅的light"
        response, debug_info = self.engine.process_input(mixed_input)
        self.assertIsNotNone(response)
    
    def test_network_error_simulation(self):
        """测试网络错误情况"""
        # 模拟API调用失败
        original_client = self.engine.api_client
        self.engine.api_client = None
        
        response, debug_info = self.engine.process_input("打开灯")
        self.assertIsNotNone(response)
        
        # 恢复
        self.engine.api_client = original_client
```

### 2. 性能测试

创建 `tests/test_performance.py`：

```python
import unittest
import time
import threading
from dialogue_manager.engine import DialogueEngine, EngineConfig

class TestPerformance(unittest.TestCase):
    
    def setUp(self):
        config = EngineConfig(enable_device_manager=True)
        self.engine = DialogueEngine(api_key="test_key", config=config)
    
    def test_response_time(self):
        """测试响应时间"""
        self.engine.start_session("perf_user")
        
        start_time = time.time()
        response, debug_info = self.engine.process_input("打开客厅的灯")
        end_time = time.time()
        
        response_time = end_time - start_time
        self.assertLess(response_time, 2.0, "响应时间应小于2秒")
        
        self.engine.end_session()
    
    def test_concurrent_sessions(self):
        """测试并发会话处理"""
        results = []
        
        def process_session(user_id):
            engine = DialogueEngine(api_key="test_key", config=EngineConfig())
            engine.start_session(f"user_{user_id}")
            
            start_time = time.time()
            response, _ = engine.process_input("打开灯")
            end_time = time.time()
            
            results.append({
                "user_id": user_id,
                "response_time": end_time - start_time,
                "success": response is not None
            })
            
            engine.end_session()
        
        # 创建10个并发会话
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_session, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertTrue(result["success"])
            self.assertLess(result["response_time"], 5.0)
```

### 3. 复杂场景测试

创建 `tests/test_complex_scenarios.py`：

```python
import unittest
from dialogue_manager.engine import DialogueEngine, EngineConfig

class TestComplexScenarios(unittest.TestCase):
    
    def setUp(self):
        config = EngineConfig(enable_device_manager=True)
        self.engine = DialogueEngine(api_key="test_key", config=config)
        self.engine.start_session("complex_user")
    
    def test_multi_device_interaction(self):
        """测试多设备交互场景"""
        # 场景：用户想要设置睡前模式
        responses = []
        
        # 第一步：关闭客厅灯
        response, _ = self.engine.process_input("关闭客厅的灯")
        responses.append(response)
        
        # 第二步：打开卧室灯并调暗
        response, _ = self.engine.process_input("打开卧室的灯")
        responses.append(response)
        
        # 第三步：使用代词调节亮度
        response, _ = self.engine.process_input("把它调暗一点")
        responses.append(response)
        
        # 第四步：设置空调
        response, _ = self.engine.process_input("把空调温度设为24度")
        responses.append(response)
        
        # 验证所有响应都成功
        for response in responses:
            self.assertIsNotNone(response)
            self.assertNotIn("抱歉", response)
    
    def test_context_switching(self):
        """测试上下文切换场景"""
        # 建立初始焦点
        self.engine.process_input("打开客厅的电视")
        self.assertEqual(self.engine.context["current_focus"]["value"], "电视")
        
        # 切换到其他设备
        self.engine.process_input("调节空调温度到26度")
        self.assertEqual(self.engine.context["current_focus"]["value"], "空调")
        
        # 使用代词应该指向新的焦点
        response, debug_info = self.engine.process_input("把它关掉")
        entities = debug_info.get("intent_result", {}).get("entities", [])
        device_entity = next((e for e in entities if e.get("entity_type") == "device"), None)
        self.assertEqual(device_entity["value"], "空调")
    
    def test_ambiguous_input_resolution(self):
        """测试模糊输入的解析"""
        # 建立上下文
        self.engine.process_input("打开客厅的灯")
        
        # 模糊的间接指令
        test_inputs = [
            "太亮了",
            "有点暗",
            "调节一下",
            "换个模式"
        ]
        
        for user_input in test_inputs:
            response, debug_info = self.engine.process_input(user_input)
            # 应该能够处理或至少给出合理的澄清
            self.assertIsNotNone(response)
```

## 🔧 错误处理和恢复机制

### 1. 异常处理增强

在 `dialogue_manager/engine.py` 中添加全局异常处理：

```python
class DialogueEngine:
    def __init__(self, api_key: str, config: EngineConfig = None):
        # ... 现有初始化代码 ...
        
        # 添加错误恢复配置
        self.error_recovery_enabled = True
        self.max_retry_attempts = 3
        self.fallback_responses = {
            "api_error": "抱歉，系统暂时无法处理您的请求，请稍后再试。",
            "parsing_error": "我没有完全理解您的意思，请您换个方式描述。",
            "device_error": "设备操作失败，请检查设备状态后重试。",
            "unknown_error": "系统出现问题，请稍后再试。"
        }
    
    def process_input_with_recovery(self, user_input: str) -> Tuple[str, Dict]:
        """带错误恢复的输入处理"""
        for attempt in range(self.max_retry_attempts):
            try:
                return self.process_input(user_input)
            except APIException as e:
                self.logger.error(f"API error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retry_attempts - 1:
                    return self.fallback_responses["api_error"], {"error": str(e)}
                time.sleep(0.5 * (attempt + 1))  # 指数退避
            except ParsingException as e:
                self.logger.error(f"Parsing error: {e}")
                return self.fallback_responses["parsing_error"], {"error": str(e)}
            except DeviceException as e:
                self.logger.error(f"Device error: {e}")
                return self.fallback_responses["device_error"], {"error": str(e)}
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retry_attempts - 1:
                    return self.fallback_responses["unknown_error"], {"error": str(e)}
        
        return self.fallback_responses["unknown_error"], {"error": "Max retries exceeded"}
```

### 2. 自定义异常类

创建 `dialogue_manager/exceptions.py`：

```python
"""对话系统自定义异常类"""

class DialogueException(Exception):
    """对话系统基础异常"""
    pass

class APIException(DialogueException):
    """API调用异常"""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code

class ParsingException(DialogueException):
    """解析异常"""
    pass

class DeviceException(DialogueException):
    """设备操作异常"""
    def __init__(self, message: str, device_id: str = None):
        super().__init__(message)
        self.device_id = device_id

class ContextException(DialogueException):
    """上下文处理异常"""
    pass
```

## 📊 监控和日志增强

### 1. 结构化日志

在 `dialogue_manager/engine.py` 中添加结构化日志：

```python
import json
import logging
from datetime import datetime

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_dialogue_turn(self, session_id: str, turn_id: int, user_input: str, 
                         system_response: str, debug_info: Dict):
        """记录对话轮次的结构化日志"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "dialogue_turn",
            "session_id": session_id,
            "turn_id": turn_id,
            "user_input": user_input,
            "system_response": system_response,
            "processing_time": debug_info.get("processing_time", 0),
            "intent_confidence": debug_info.get("intent_result", {}).get("confidence", 0),
            "api_calls_count": len(debug_info.get("api_calls", [])),
            "focus_entity": debug_info.get("context_updates", {}).get("current_focus")
        }
        self.logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def log_error(self, session_id: str, error_type: str, error_message: str, context: Dict):
        """记录错误的结构化日志"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "error",
            "session_id": session_id,
            "error_type": error_type,
            "error_message": error_message,
            "context": context
        }
        self.logger.error(json.dumps(log_data, ensure_ascii=False))
```

### 2. 性能监控

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "api_call_count": 0,
            "focus_switches": 0
        }
    
    def record_request(self, success: bool, response_time: float, api_calls: int, focus_switched: bool):
        """记录请求指标"""
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # 更新平均响应时间
        current_avg = self.metrics["average_response_time"]
        total = self.metrics["total_requests"]
        self.metrics["average_response_time"] = (current_avg * (total - 1) + response_time) / total
        
        self.metrics["api_call_count"] += api_calls
        if focus_switched:
            self.metrics["focus_switches"] += 1
    
    def get_metrics(self) -> Dict:
        """获取当前指标"""
        return self.metrics.copy()
```

## 🚀 部署和配置优化

### 1. 配置管理增强

创建 `config/default.yaml`：

```yaml
# 对话引擎配置
engine:
  max_turns: 10
  confidence_threshold: 0.7
  session_timeout: 3600
  execution_mode: "internal_first"  # internal_first | llm_first | parallel
  focus_entity_turn_decay: 2
  
# API配置
api:
  model_name: "deepseek-chat"
  timeout: 30
  max_retries: 3
  retry_delay: 0.5
  
# 设备管理
device_manager:
  enabled: true
  default_room: "客厅"
  device_timeout: 5
  
# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/dialogue_system.log"
  max_size: "10MB"
  backup_count: 5
  
# 监控配置
monitoring:
  enabled: true
  metrics_interval: 60
  alert_thresholds:
    error_rate: 0.1
    response_time: 2.0
```

### 2. 配置加载器

创建 `dialogue_manager/config_loader.py`：

```python
import yaml
import os
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
    
    def load_config(self, env: str = "default") -> Dict[str, Any]:
        """加载配置文件"""
        config_file = self.config_dir / f"{env}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 环境变量覆盖
        config = self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        # API Key从环境变量获取
        if "DIALOGUE_API_KEY" in os.environ:
            config.setdefault("api", {})["key"] = os.environ["DIALOGUE_API_KEY"]
        
        # 日志级别覆盖
        if "LOG_LEVEL" in os.environ:
            config.setdefault("logging", {})["level"] = os.environ["LOG_LEVEL"]
        
        return config
```

## 📈 分阶段实施路线图

### 🎯 里程碑 M1：P0 核心可用性（1周内完成）

**目标**：不依赖LLM即能稳定处理省略指令和焦点切换

**任务清单**：
1. **省略消解功能**（3天）
   - 在 `intent.py` 实现 `_extract_entities_with_context`
   - 替换 `recognize` 方法中的实体抽取入口
   - 编写单元测试覆盖主要场景

2. **智能焦点切换**（2天）
   - 在 `engine.py` 实现 `_should_switch_focus`
   - 接入 `_update_context` 方法
   - 添加配置开关和测试用例

**验收标准**：
- 省略主语指令成功率 > 85%
- 焦点切换符合预期策略
- 通过回归测试，无功能倒退

**风险控制**：
- 保留feature flags，可一键回退
- 灰度发布，先覆盖10%会话

### 🤖 里程碑 M2：P1 智能回退（1周）

**目标**：在规则低置信时启用LLM最小回退

**任务清单**：
1. **LLM回退实现**（4天）
   - 实现 `_get_intent_from_llm` 等方法
   - 构建智能家居领域提示模板
   - 添加JSON解析和验证逻辑

2. **成本与超时控制**（1天）
   - 配置预算阈值和超时机制
   - 实现灰度开关和采样策略

**验收标准**：
- 回退触发率 < 20%，有效纠正率 > 70%
- P95响应时延 ≤ 2.5s
- 成本在预算范围内

**风险控制**：
- 预算双闸，超预算自动禁用
- 分阶段灰度：5% → 10% → 25%

### 🛡️ 里程碑 M3：P2-P3 稳定性增强（1周）

**目标**：完善错误处理与可观测性

**任务清单**：
1. **异常处理**（2天）
   - 创建自定义异常类体系
   - 实现幂等性控制机制
   - 添加用户友好的错误提示

2. **结构化日志**（2天）
   - 实现关键字段的结构化记录
   - 添加性能指标埋点
   - 配置PII脱敏处理

3. **监控与报警**（1天）
   - 设置关键指标阈值
   - 配置异常情况报警

**验收标准**：
- 系统错误率 ≤ 1%
- 关键路径可追溯
- 性能指标可量化

### 🚀 里程碑 M4+：P4-P5 扩展优化（按需推进）

**包含功能**：
- 并发处理与性能优化
- 多语言与词法归一化
- 上下文持久化
- 测试工程化与CI守护

**推进策略**：根据业务压力和用户反馈确定优先级

## 🎯 成功指标与配置建议

### 📊 分阶段目标指标

**P0阶段指标**：
- 省略主语指令识别准确率 > 85%
- 焦点切换准确性 > 90%
- 系统稳定性（无崩溃）> 99.5%

**P1阶段指标**：
- LLM回退触发率 < 20%，有效纠正率 > 70%
- P95响应时延：无LLM ≤ 800ms，回退场景 ≤ 2.5s
- 识别准确率较P0提升 > 5%

**P2-P3阶段指标**：
- 系统错误率 ≤ 1%
- JSON解析失败回退率 < 2%
- 关键路径可追溯率 = 100%

### ⚙️ 关键配置参数

```yaml
# 推荐初始配置
engine:
  min_confidence_for_llm: 0.4
  llm_fallback_enabled: true  # 灰度开启
  llm_timeout: 3  # 秒
  per_turn_max_tokens: 500
  per_session_budget: 1000
  enable_context_entity_fill: true
  focus_switch_policy: "conservative"
  
monitoring:
  error_rate_threshold: 0.01
  response_time_p95_threshold: 2500  # ms
  llm_fallback_rate_threshold: 0.2
```

### 🛡️ 风险控制策略

**技术风险**：
- Feature flags支持一键回退到旧逻辑
- 预算阈值与超时双重保护
- 灰度发布：10% → 25% → 50% → 100%

**业务风险**：
- 误操作风险：低置信度场景增加澄清确认
- 成本风险：单日预算上限，超限自动降级
- 性能风险：P95延迟监控，异常时自动禁用回退

**回滚策略**：
- 保守启发式策略，优先保证不出错
- 黄金样本集持续验证，发现回退立即修正
- 关键指标异常时自动触发降级模式

## 📚 参考资源

### 开发工具
- **测试框架**：pytest, unittest
- **代码质量**：pylint, black, mypy
- **监控工具**：prometheus, grafana
- **日志分析**：ELK Stack

### 相关文档
- [Python异步编程最佳实践](https://docs.python.org/3/library/asyncio.html)
- [对话系统设计模式](https://rasa.com/docs/)
- [LLM集成指南](https://platform.openai.com/docs/)
- [系统监控最佳实践](https://prometheus.io/docs/)

---

## 🚀 下一步执行建议

### 立即可执行（本周内）
1. **启动P0核心功能**：
   - 在 `intent.py` 实现 `_extract_entities_with_context` 方法
   - 在 `engine.py` 实现 `_should_switch_focus` 方法
   - 补齐最小单元测试集

2. **准备P1回退机制**：
   - 在引擎配置中预留LLM回退参数和开关
   - 在 `process_input` 中预留低置信度触发钩子（暂不启用）

### 验证与迭代
- 构建黄金样本集（省略主语、代词指代、多轮焦点继承等场景）
- 设置CI回归基线，防止功能倒退
- 灰度发布，持续监控关键指标

### 模型型号备注
- **当前使用模型**：zai-org/GLM-4.5-Air（SiliconFlow）
- **配置位置**：`dialogue_manager/api_client.py` 中的 `model_id`
- **优化建议**：后续可根据成本/时延需求在同接口层做可配置切换

---

**重要提醒**：本改进指南采用"搭积木式"分阶段实施策略，避免一次性大规模重构。建议严格按照P0→P1→P2→P3的优先级顺序推进，确保每个阶段都有明确的交付物和验收标准。在实施过程中，请根据实际业务场景调整具体的实现细节和参数配置。