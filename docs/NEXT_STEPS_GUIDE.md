# 对话系统下一步开发操作指南

## 📋 文档概述

本文档为程序员提供明确的下一步开发任务指导，基于当前项目状态（P0核心功能已完成，102个测试全部通过），详细说明P1智能回退机制的实施步骤。

## 🎯 1. 任务目标

### 主要目标
实施P1智能回退机制，在规则引擎低置信度时启用LLM回退，提升复杂/模糊指令的识别准确率。

### 具体目标
- 实现LLM意图识别回退功能
- 建立成本控制和超时机制
- 保持系统稳定性和性能
- 通过测试验证功能正确性

### 成功指标
- LLM回退触发率 < 20%
- 有效纠正率 > 70%
- P95响应时延 ≤ 2.5s
- 识别准确率较P0提升 > 5%

## 📝 2. 分步骤操作指南

### 阶段一：核心功能实现（预计4天）

#### 步骤1：添加LLM回退配置参数

**操作位置**：`dialogue_manager/engine.py` - `EngineConfig` 类

**具体操作**：
```python
# 在 EngineConfig 类中添加以下配置项
min_confidence_for_llm: float = 0.4  # LLM回退触发阈值
llm_fallback_enabled: bool = True    # LLM回退开关
llm_timeout: int = 3                 # LLM调用超时时间（秒）
per_turn_max_tokens: int = 500       # 单轮最大token数
per_session_budget: int = 1000       # 单会话预算限制
```

**验证方法**：
- 运行现有测试确保配置加载正常
- 检查配置项是否正确初始化

#### 步骤2：实现LLM意图识别方法

**操作位置**：`dialogue_manager/engine.py` - `DialogueEngine` 类

**具体操作**：
1. 添加 `_get_intent_from_llm` 方法
2. 添加 `_build_llm_intent_prompt` 方法
3. 添加 `_parse_llm_intent_response` 方法
4. 添加 `_create_unknown_intent_result` 方法

**代码模板**：
```python
def _get_intent_from_llm(self, user_input: str, context: Dict, debug_info: Dict) -> Dict:
    """使用LLM进行意图识别的回退机制"""
    try:
        # 构建LLM提示
        prompt = self._build_llm_intent_prompt(user_input, context)
        
        # 调用LLM API
        api_response = self.api_client.chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=self.config.per_turn_max_tokens
        )
        
        # 记录API调用
        debug_info["api_calls"].append({
            "success": api_response.success,
            "content": api_response.content,
            "error": api_response.error_message,
            "response_time": api_response.response_time,
            "request": {
                "prompt": prompt, 
                "model": self.api_client.model_id, 
                "purpose": "intent_fallback"
            },
            "response": api_response.raw_response,
        })
        
        if api_response.success and api_response.content:
            return self._parse_llm_intent_response(api_response.content)
        else:
            self.logger.error(f"LLM intent fallback failed: {api_response.error_message}")
            return self._create_unknown_intent_result(user_input)
            
    except Exception as e:
        self.logger.error(f"LLM intent fallback error: {e}")
        return self._create_unknown_intent_result(user_input)
```

**验证方法**：
- 单独测试每个方法的功能
- 验证API调用和响应解析
- 测试异常处理逻辑

#### 步骤3：集成LLM回退到主流程

**操作位置**：`dialogue_manager/engine.py` - `process_input` 方法

**具体操作**：
在意图识别后添加回退逻辑：
```python
# 在 process_input 方法中，意图识别后添加：
if (self.config.llm_fallback_enabled and 
    intent_result["confidence"] < self.config.min_confidence_for_llm):
    
    self.logger.info(f"Low confidence ({intent_result['confidence']}), trying LLM fallback")
    llm_result = self._get_intent_from_llm(user_input, self.context, debug_info)
    
    if llm_result["confidence"] > intent_result["confidence"]:
        intent_result = llm_result
        debug_info["llm_fallback_used"] = True
        self.logger.info(f"LLM fallback improved confidence: {llm_result['confidence']}")
```

**验证方法**：
- 测试低置信度场景触发回退
- 验证高置信度场景不触发回退
- 检查debug信息记录正确

### 阶段二：成本控制和监控（预计1天）

#### 步骤4：实现预算控制机制

**操作位置**：`dialogue_manager/engine.py` - `DialogueEngine` 类

**具体操作**：
1. 添加会话级token计数器
2. 实现预算检查逻辑
3. 添加超预算自动禁用机制

**代码模板**：
```python
def __init__(self, api_key: str, config: Optional[EngineConfig] = None):
    # ... 现有初始化代码 ...
    self.session_token_usage = 0  # 会话级token使用量
    self.llm_fallback_disabled = False  # 预算超限标志

def _check_budget_limit(self) -> bool:
    """检查是否超出预算限制"""
    if self.session_token_usage >= self.config.per_session_budget:
        if not self.llm_fallback_disabled:
            self.logger.warning(f"Session budget exceeded: {self.session_token_usage}")
            self.llm_fallback_disabled = True
        return False
    return True
```

**验证方法**：
- 模拟高token使用场景
- 验证预算超限时自动禁用
- 测试预算重置机制

#### 步骤5：添加性能监控

**操作位置**：`dialogue_manager/engine.py`

**具体操作**：
1. 记录LLM调用次数和耗时
2. 统计回退触发率和成功率
3. 监控响应时间分布

**验证方法**：
- 检查监控数据准确性
- 验证性能指标计算正确

### 阶段三：测试和验证（预计1天）

#### 步骤6：编写单元测试

**操作位置**：`tests/test_llm_fallback.py`（新建文件）

**测试用例**：
1. 测试LLM回退触发条件
2. 测试API调用成功和失败场景
3. 测试JSON解析和验证
4. 测试预算控制机制
5. 测试超时处理

**代码模板**：
```python
import unittest
from unittest.mock import Mock, patch
from dialogue_manager.engine import DialogueEngine, EngineConfig

class TestLLMFallback(unittest.TestCase):
    
    def setUp(self):
        config = EngineConfig(
            llm_fallback_enabled=True,
            min_confidence_for_llm=0.4,
            llm_timeout=3
        )
        self.engine = DialogueEngine("test_key", config)
        self.engine.start_session("test_user")
    
    def test_llm_fallback_trigger(self):
        """测试LLM回退触发条件"""
        # 模拟低置信度意图识别结果
        with patch.object(self.engine.intent_recognizer, 'recognize') as mock_recognize:
            mock_recognize.return_value = {
                "intent": "unknown",
                "confidence": 0.3,  # 低于阈值
                "entities": [],
                "need_clarification": True
            }
            
            with patch.object(self.engine, '_get_intent_from_llm') as mock_llm:
                mock_llm.return_value = {
                    "intent": "device_control",
                    "confidence": 0.8,
                    "entities": [{"entity_type": "device", "value": "灯"}]
                }
                
                response, debug_info = self.engine.process_input("模糊的指令")
                
                # 验证LLM回退被调用
                mock_llm.assert_called_once()
                self.assertTrue(debug_info.get("llm_fallback_used", False))
```

#### 步骤7：集成测试

**操作**：
1. 运行完整测试套件
2. 验证新功能不影响现有功能
3. 测试端到端场景

**命令**：
```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定的LLM回退测试
python -m pytest tests/test_llm_fallback.py -v

# 运行性能测试
python -m pytest tests/test_performance.py -v
```

## 🎯 3. 预期输出结果

### 功能输出
1. **LLM回退功能**：低置信度时自动触发LLM意图识别
2. **成本控制**：预算超限时自动禁用回退功能
3. **监控数据**：详细的性能和使用统计
4. **调试信息**：完整的API调用和决策链记录

### 代码输出
1. **新增方法**：4个LLM相关方法
2. **配置扩展**：5个新配置参数
3. **测试用例**：至少10个测试场景
4. **文档更新**：API文档和配置说明

### 性能输出
1. **响应时间**：P95 ≤ 2.5s（包含LLM调用）
2. **准确率提升**：复杂指令识别率提升5%+
3. **稳定性**：测试通过率保持100%

## ✅ 4. 完成标准

### 功能完成标准
- [ ] LLM回退机制正常工作
- [ ] 预算控制机制有效
- [ ] 超时处理正确
- [ ] 异常处理完善
- [ ] 配置参数生效

### 质量完成标准
- [ ] 所有新增代码有单元测试覆盖
- [ ] 测试通过率100%
- [ ] 代码符合项目规范
- [ ] 性能指标达标
- [ ] 文档更新完整

### 验收测试清单
```bash
# 1. 基础功能测试
python -c "from dialogue_manager.engine import DialogueEngine, EngineConfig; print('Import OK')"

# 2. 配置测试
python -c "from dialogue_manager.engine import EngineConfig; c=EngineConfig(); print(f'LLM enabled: {c.llm_fallback_enabled}')"

# 3. 完整测试套件
python -m pytest tests/ -v --tb=short

# 4. 性能基准测试
python -m pytest tests/test_performance.py::TestPerformance::test_response_time -v

# 5. LLM回退功能测试
python debug_llm_fallback.py  # 需要创建此调试脚本
```

## 🔧 5. 问题排查指引

### 常见问题及解决方案

#### 问题1：LLM回退不触发
**症状**：低置信度场景下LLM回退未被调用

**排查步骤**：
1. 检查 `llm_fallback_enabled` 配置是否为 `True`
2. 验证置信度是否低于 `min_confidence_for_llm` 阈值
3. 检查预算是否已超限（`llm_fallback_disabled` 标志）
4. 查看日志中的置信度和触发条件记录

**解决方案**：
```python
# 调试代码
print(f"Config enabled: {engine.config.llm_fallback_enabled}")
print(f"Confidence: {intent_result['confidence']} vs threshold: {engine.config.min_confidence_for_llm}")
print(f"Budget disabled: {engine.llm_fallback_disabled}")
```

#### 问题2：API调用失败
**症状**：LLM API调用返回错误或超时

**排查步骤**：
1. 检查API密钥是否有效
2. 验证网络连接
3. 检查API配额和限制
4. 查看API响应错误信息

**解决方案**：
```python
# 测试API连接
from dialogue_manager.api_client import SiliconFlowClient
client = SiliconFlowClient("your_api_key")
response = client.chat_completion([{"role": "user", "content": "test"}])
print(f"API test result: {response.success}, {response.error_message}")
```

#### 问题3：JSON解析失败
**症状**：LLM返回的内容无法解析为有效JSON

**排查步骤**：
1. 检查LLM提示是否明确要求JSON格式
2. 验证返回内容的格式
3. 检查必要字段是否存在

**解决方案**：
- 优化提示模板，强调JSON格式要求
- 添加更严格的JSON验证
- 实现降级处理机制

#### 问题4：性能问题
**症状**：响应时间过长或超时

**排查步骤**：
1. 检查LLM调用耗时
2. 验证超时配置
3. 分析并发处理情况

**解决方案**：
- 调整 `llm_timeout` 配置
- 优化提示长度
- 实现异步处理（后续版本）

### 调试工具和命令

#### 创建调试脚本
**文件**：`debug_llm_fallback.py`
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dialogue_manager.engine import DialogueEngine, EngineConfig

def debug_llm_fallback():
    """调试LLM回退功能"""
    config = EngineConfig(
        llm_fallback_enabled=True,
        min_confidence_for_llm=0.4,
        enable_device_manager=True
    )
    
    engine = DialogueEngine("your_api_key_here", config)
    engine.start_session('debug_user')
    
    # 测试低置信度输入
    test_inputs = [
        "模糊的指令",
        "不太清楚的要求",
        "复杂的表达"
    ]
    
    for user_input in test_inputs:
        print(f"\n=== 测试输入: '{user_input}' ===")
        response, debug_info = engine.process_input(user_input)
        
        print(f"响应: {response}")
        print(f"LLM回退使用: {debug_info.get('llm_fallback_used', False)}")
        print(f"置信度: {debug_info['intent_result']['confidence']}")
        
        if 'api_calls' in debug_info:
            print(f"API调用次数: {len(debug_info['api_calls'])}")

if __name__ == "__main__":
    debug_llm_fallback()
```

#### 日志配置
```python
# 在engine.py中添加详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 在关键位置添加日志
logger.debug(f"Intent confidence: {intent_result['confidence']}")
logger.info(f"LLM fallback triggered for input: {user_input}")
logger.error(f"LLM API call failed: {error_message}")
```

## 📚 6. 相关参考资料

### 项目文档
- [改进指南主文档](./dialogue_system_improvement_guide.md)
- [API客户端文档](./dialogue_manager/api_client.py)
- [意图识别文档](./dialogue_manager/intent.py)
- [引擎核心文档](./dialogue_manager/engine.py)

### 测试参考
- [现有测试用例](./tests/)
- [设备管理器测试](./tests/test_device_manager.py)
- [对话焦点测试](./tests/test_dialogue_focus.py)
- [单元测试框架](./tests/test_unit.py)

### 技术文档
- [SiliconFlow API文档](https://docs.siliconflow.cn/)
- [Python异步编程](https://docs.python.org/3/library/asyncio.html)
- [pytest测试框架](https://docs.pytest.org/)
- [JSON Schema验证](https://json-schema.org/)

### 配置示例
```yaml
# config/development.yaml
engine:
  min_confidence_for_llm: 0.4
  llm_fallback_enabled: true
  llm_timeout: 3
  per_turn_max_tokens: 500
  per_session_budget: 1000
  enable_context_entity_fill: true
  focus_switch_policy: "conservative"
  
api:
  model_name: "deepseek-chat"
  timeout: 30
  max_retries: 3
  
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 📋 7. 进度追踪清单

### 开发进度
- [ ] 步骤1：添加LLM回退配置参数
- [ ] 步骤2：实现LLM意图识别方法
- [ ] 步骤3：集成LLM回退到主流程
- [ ] 步骤4：实现预算控制机制
- [ ] 步骤5：添加性能监控
- [ ] 步骤6：编写单元测试
- [ ] 步骤7：集成测试

### 验收清单
- [ ] 功能完成标准（5项）
- [ ] 质量完成标准（5项）
- [ ] 验收测试清单（5项）

### 文档更新
- [ ] API文档更新
- [ ] 配置说明更新
- [ ] 使用示例更新
- [ ] 故障排查指南

---

**注意事项**：
1. 在开始实施前，请确保当前代码库是最新版本
2. 建议创建feature分支进行开发：`git checkout -b feature/llm-fallback`
3. 每完成一个步骤后，运行测试确保功能正常
4. 遇到问题时，优先查看日志和调试信息
5. 完成后记得更新相关文档和示例代码

**预计完成时间**：6个工作日
**风险等级**：中等
**依赖项**：SiliconFlow API访问权限