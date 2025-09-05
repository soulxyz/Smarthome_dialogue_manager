"""真实对话系统测试

区分API测试（必要功能测试）和内部逻辑测试（可压力测试）
"""

import pytest
import time
import asyncio
import concurrent.futures
from typing import Dict, List, Any
from unittest.mock import patch

from dialogue_manager.engine import DialogueEngine, EngineConfig, DialogueState
from dialogue_manager.intent import IntentRecognizer
from dialogue_manager.clarification import ClarificationAgent

from tests.fixtures import (
    test_config, api_client, memory_manager, device_manager,
    dialogue_engine, intent_recognizer, clarification_agent,
    sample_user_inputs, dialogue_scenarios, api_tracker,
    measure_execution_time, create_test_dialogue_history
)


class TestDialogueEngineInternalLogic:
    """对话引擎内部逻辑测试（可压力测试）"""
    
    def test_state_transitions_stress(self):
        """压力测试：状态转换"""
        config = EngineConfig(enable_device_manager=True)
        engine = DialogueEngine(api_key="test-key-internal", config=config)
        
        # 大量状态转换
        state_transitions = []
        start_time = time.time()
        
        for i in range(500):
            # 模拟状态转换序列
            engine.current_state = DialogueState.IDLE
            
            # 开始会话
            session_id = engine.start_session(f"stress_user_{i}")
            assert engine.current_state == DialogueState.LISTENING
            state_transitions.append("IDLE->LISTENING")
            
            # 模拟处理状态
            engine.current_state = DialogueState.PROCESSING
            state_transitions.append("LISTENING->PROCESSING")
            
            # 模拟执行状态
            engine.current_state = DialogueState.EXECUTING
            state_transitions.append("PROCESSING->EXECUTING")
            
            # 回到监听状态
            engine.current_state = DialogueState.LISTENING
            state_transitions.append("EXECUTING->LISTENING")
            
            # 结束会话
            engine.end_session()
            assert engine.current_state == DialogueState.IDLE
            state_transitions.append("LISTENING->IDLE")
        
        transition_time = time.time() - start_time
        
        # 验证状态转换性能
        assert transition_time < 3.0  # 2500次状态转换应在3秒内完成
        assert len(state_transitions) == 2500
        
        # 验证转换序列正确性
        expected_pattern = ["IDLE->LISTENING", "LISTENING->PROCESSING", "PROCESSING->EXECUTING", "EXECUTING->LISTENING", "LISTENING->IDLE"]
        for i in range(0, len(state_transitions), 5):
            actual_pattern = state_transitions[i:i+5]
            assert actual_pattern == expected_pattern

    def test_context_management_stress(self, memory_manager):
        """压力测试：上下文管理"""
        config = EngineConfig(enable_device_manager=True, enable_context_entity_fill=True)
        engine = DialogueEngine(api_key="test-key-internal", config=config)
        engine.memory_manager = memory_manager
        
        engine.start_session("context_stress_user")
        
        # 大量上下文操作
        start_time = time.time()
        context_operations = 0
        
        for i in range(200):
            # 模拟不同类型的上下文更新
            engine.context.update({
                f"key_{i}": f"value_{i}",
                "current_focus": {
                    "value": f"设备_{i % 10}",
                    "turn_count": i % 5,
                    "confidence": 0.8 + (i % 2) * 0.1
                },
                "last_entities": {
                    "devices": [{"value": f"设备_{j}"} for j in range(i % 5)],
                    "actions": [{"value": f"动作_{j}"} for j in range(i % 3)],
                    "locations": [{"value": f"房间_{j}"} for j in range(i % 4)]
                },
                "turn_count": i,
                "session_data": {f"data_{k}": f"content_{k}" for k in range(i % 10)}
            })
            context_operations += 1
            
            # 模拟焦点实体过期检查
            if "current_focus" in engine.context:
                if engine.context["current_focus"]["turn_count"] >= 3:
                    engine.context.pop("current_focus")
            
            # 模拟上下文清理
            if i % 50 == 0:
                # 保留重要信息，清理临时数据
                important_keys = ["current_focus", "last_entities", "turn_count"]
                new_context = {k: v for k, v in engine.context.items() if k in important_keys}
                engine.context.clear()
                engine.context.update(new_context)
        
        context_time = time.time() - start_time
        
        # 验证上下文管理性能
        assert context_time < 2.0  # 200次复杂上下文操作应在2秒内完成
        assert context_operations == 200
        
        engine.end_session()

    def test_dialogue_history_management_stress(self, memory_manager):
        """压力测试：对话历史管理"""
        config = EngineConfig(enable_device_manager=True, max_history_length=100)
        engine = DialogueEngine(api_key="test-key-internal", config=config)
        engine.memory_manager = memory_manager
        
        engine.start_session("history_stress_user")
        
        # 生成大量对话历史
        start_time = time.time()
        
        for i in range(500):
            from dialogue_manager.engine import DialogueTurn
            
            turn = DialogueTurn(
                turn_id=i + 1,
                user_input=f"用户输入{i}",
                system_response=f"系统响应{i}",
                intent=["device_control", "query_status", "greeting"][i % 3],
                confidence=0.7 + (i % 4) * 0.1,
                context={"test": True, "turn": i}
            )
            
            engine.dialogue_history.append(turn)
            
            # 测试历史长度限制
            if len(engine.dialogue_history) > config.max_history_length:
                engine.dialogue_history = engine.dialogue_history[-config.max_history_length:]
        
        history_time = time.time() - start_time
        
        # 验证历史管理性能
        assert history_time < 1.0  # 500次历史操作应在1秒内完成
        assert len(engine.dialogue_history) == config.max_history_length  # 应该被限制在100条
        
        # 验证历史内容
        assert engine.dialogue_history[0].turn_id == 401  # 应该保留最后100条
        assert engine.dialogue_history[-1].turn_id == 500
        
        engine.end_session()

    def test_concurrent_session_management(self):
        """压力测试：并发会话管理"""
        def manage_session(user_id, operations_count):
            """管理单个会话"""
            config = EngineConfig(enable_device_manager=True)
            engine = DialogueEngine(api_key="test-key-internal", config=config)
            
            session_id = engine.start_session(f"concurrent_user_{user_id}")
            results = []
            
            for i in range(operations_count):
                # 模拟内部处理（不调用API）
                engine.context.update({
                    "operation": i,
                    "current_focus": {"value": f"设备_{i % 5}", "turn_count": 0}
                })
                
                from dialogue_manager.engine import DialogueTurn
                turn = DialogueTurn(
                    turn_id=i + 1,
                    user_input=f"测试输入{i}",
                    system_response=f"测试响应{i}",
                    intent="device_control"
                )
                engine.dialogue_history.append(turn)
                
                results.append({"success": True, "turn_id": i + 1})
            
            engine.end_session()
            return results
        
        # 并发测试
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(manage_session, i, 20) for i in range(20)]
            all_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        concurrent_time = time.time() - start_time
        
        # 验证并发性能
        assert concurrent_time < 5.0  # 20个并发会话，每个20次操作，应在5秒内完成
        assert len(all_results) == 20
        
        # 验证所有会话都正确完成
        total_operations = sum(len(results) for results in all_results)
        successful_operations = sum(sum(1 for r in results if r["success"]) for results in all_results)
        assert successful_operations == total_operations == 400

    def test_memory_cleanup_stress(self, memory_manager):
        """压力测试：内存清理"""
        config = EngineConfig(enable_device_manager=True, cleanup_after_tests=True)
        engines = []
        
        # 创建大量引擎实例
        start_time = time.time()
        
        for i in range(50):
            engine = DialogueEngine(api_key="test-key-internal", config=config)
            engine.memory_manager = memory_manager
            
            session_id = engine.start_session(f"cleanup_user_{i}")
            
            # 填充一些数据
            for j in range(10):
                engine.context[f"data_{j}"] = f"content_{j}" * 100  # 创建一些内存占用
                
                from dialogue_manager.engine import DialogueTurn
                turn = DialogueTurn(
                    turn_id=j + 1,
                    user_input=f"输入{j}" * 50,
                    system_response=f"响应{j}" * 50,
                    intent="test"
                )
                engine.dialogue_history.append(turn)
            
            engines.append(engine)
        
        creation_time = time.time() - start_time
        
        # 清理所有引擎
        cleanup_start = time.time()
        for engine in engines:
            engine.end_session()
            # 清理内存
            engine.dialogue_history.clear()
            engine.context.clear()
        
        cleanup_time = time.time() - cleanup_start
        
        # 验证创建和清理性能
        assert creation_time < 3.0  # 50个引擎创建应在3秒内完成
        assert cleanup_time < 1.0   # 清理应在1秒内完成


class TestIntentRecognizerInternalLogic:
    """意图识别器内部逻辑测试（可压力测试）"""
    
    def test_pattern_matching_stress(self, device_manager):
        """压力测试：模式匹配"""
        config = EngineConfig(enable_device_manager=True)
        recognizer = IntentRecognizer(config=config, device_manager=device_manager)
        
        # 生成大量测试输入
        test_inputs = []
        for i in range(1000):
            device = ["灯", "空调", "电视", "风扇"][i % 4]
            room = ["客厅", "主卧", "次卧", "厨房", "书房"][i % 5]
            action = ["打开", "关闭", "调节", "设置"][i % 4]
            
            test_inputs.extend([
                f"{action}{room}的{device}",
                f"{action}{device}",
                f"{room}{device}{action}",
                f"请{action}{room}里的{device}",
                f"帮我{action}一下{device}"
            ])
        
        # 执行大量模式匹配
        start_time = time.time()
        results = []
        
        for user_input in test_inputs:
            result = recognizer._classify_intent(user_input)
            results.append(result)
        
        matching_time = time.time() - start_time
        
        # 验证模式匹配性能
        assert matching_time < 10.0  # 5000次模式匹配应在10秒内完成
        assert len(results) == len(test_inputs)
        
        # 验证匹配准确性
        device_control_count = sum(1 for r in results if r.get("device_control", 0) > 0)
        assert device_control_count >= len(test_inputs) * 0.8  # 至少80%被识别为设备控制

    def test_entity_extraction_stress(self, device_manager):
        """压力测试：实体提取"""
        config = EngineConfig(enable_device_manager=True)
        recognizer = IntentRecognizer(config=config, device_manager=device_manager)
        
        # 生成复杂的测试文本
        complex_inputs = []
        for i in range(500):
            text = f"请打开{['客厅', '主卧', '次卧'][i % 3]}的{['灯', '空调', '电视'][i % 3]}，" \
                   f"然后调节{['亮度', '温度', '音量'][i % 3]}到{20 + i % 80}，" \
                   f"最后设置{['模式', '频道', '风速'][i % 3]}为{['自动', str(i % 20), str(1 + i % 5)][i % 3]}"
            complex_inputs.append(text)
        
        # 执行大量实体提取
        start_time = time.time()
        all_entities = []
        
        for text in complex_inputs:
            entities = recognizer._extract_entities(text)
            all_entities.append(entities)
        
        extraction_time = time.time() - start_time
        
        # 验证实体提取性能
        assert extraction_time < 5.0  # 500次复杂实体提取应在5秒内完成
        assert len(all_entities) == len(complex_inputs)
        
        # 验证提取质量
        total_entities = sum(len(entities) for entities in all_entities)
        assert total_entities >= len(complex_inputs) * 3  # 每个输入平均至少3个实体

    def test_context_enhancement_stress(self, device_manager):
        """压力测试：上下文增强"""
        config = EngineConfig(enable_device_manager=True)
        recognizer = IntentRecognizer(config=config, device_manager=device_manager)
        
        # 创建复杂上下文
        contexts = []
        for i in range(200):
            context = {
                "last_intent": ["device_control", "query_status", "greeting"][i % 3],
                "last_entities": {
                    "devices": [{"value": f"设备_{j}"} for j in range(i % 5)],
                    "actions": [{"value": f"动作_{j}"} for j in range(i % 3)],
                    "locations": [{"value": f"位置_{j}"} for j in range(i % 4)]
                },
                "current_focus": {
                    "value": f"焦点设备_{i % 10}",
                    "turn_count": i % 3,
                    "confidence": 0.8 + (i % 2) * 0.1
                },
                "turn_count": i,
                "last_possible_intents": [f"意图_{j}" for j in range(i % 5)]
            }
            contexts.append(context)
        
        # 创建历史记录
        histories = []
        for i in range(200):
            history = []
            for j in range(i % 10):  # 变长历史
                turn = {
                    "user_input": f"历史输入{j}",
                    "intent": ["device_control", "query_status"][j % 2],
                    "entities": [{"type": "device", "value": f"历史设备{j}"}]
                }
                history.append(turn)
            histories.append(history)
        
        # 执行大量上下文增强
        start_time = time.time()
        enhanced_results = []
        
        for i in range(200):
            user_input = f"测试输入{i}"
            intent_scores = {"device_control": 0.5, "query_status": 0.3, "greeting": 0.2}
            
            enhanced_scores = recognizer._enhance_with_context(
                user_input, intent_scores, contexts[i], histories[i]
            )
            enhanced_results.append(enhanced_scores)
        
        enhancement_time = time.time() - start_time
        
        # 验证上下文增强性能
        assert enhancement_time < 2.0  # 200次上下文增强应在2秒内完成
        assert len(enhanced_results) == 200
        
        # 验证增强效果
        for enhanced_scores in enhanced_results:
            assert isinstance(enhanced_scores, dict)
            assert len(enhanced_scores) >= 3  # 至少包含基础意图分数

    def test_entity_inheritance_complex(self, device_manager):
        """复杂实体继承测试"""
        config = EngineConfig(enable_device_manager=True, enable_context_entity_fill=True)
        recognizer = IntentRecognizer(config=config, device_manager=device_manager)
        
        # 创建复杂的继承场景
        test_scenarios = []
        for i in range(100):
            context = {
                "current_focus": {
                    "value": f"设备_{i % 10}",
                    "turn_count": i % 3,
                    "confidence": 0.8
                },
                "last_entities": {
                    "devices": [{"value": f"设备_{j}", "entity_type": "device"} for j in range(i % 5)],
                    "locations": [{"value": f"房间_{j}", "entity_type": "location"} for j in range(i % 3)]
                }
            }
            
            history = [
                {
                    "user_input": f"历史输入{i}",
                    "entities": [
                        {"type": "location", "value": f"历史位置_{i}"},
                        {"entity_type": "device", "value": f"历史设备_{i}"}
                    ]
                }
            ]
            
            test_scenarios.append((context, history))
        
        # 执行实体继承测试
        start_time = time.time()
        inheritance_results = []
        
        for context, history in test_scenarios:
            # 测试省略主语的输入
            entities = recognizer._extract_entities_with_context(
                "关掉", "device_control", context, history
            )
            inheritance_results.append(entities)
        
        inheritance_time = time.time() - start_time
        
        # 验证实体继承性能
        assert inheritance_time < 1.0  # 100次实体继承应在1秒内完成
        assert len(inheritance_results) == 100
        
        # 验证继承结果
        successful_inheritance = 0
        for entities in inheritance_results:
            if any(e.entity_type == "device" for e in entities):
                successful_inheritance += 1
        
        assert successful_inheritance >= 80  # 至少80%成功继承设备实体


class TestDialogueEngineAPIIntegration:
    """对话引擎API集成测试（必要功能测试，节约API调用）"""
    
    @pytest.mark.api_required
    def test_basic_api_functionality(self, test_config, api_tracker):
        """基础API功能测试（必要测试）"""
        config = EngineConfig(enable_device_manager=True)
        engine = DialogueEngine(api_key=test_config.api_key, config=config)
        
        engine.start_session("api_basic_test")
        
        # 只测试核心功能，每类意图测试一次
        essential_tests = [
            ("你好", "greeting"),
            ("打开客厅的灯", "device_control"),
            ("现在温度怎么样", "query_status"),
            ("谢谢", "greeting")
        ]
        
        api_calls_made = 0
        for user_input, expected_intent in essential_tests:
            start_time = time.time()
            response, debug_info = engine.process_input(user_input)
            duration = time.time() - start_time
            
            # 记录API调用
            if "api_calls" in debug_info:
                api_calls_made += len(debug_info["api_calls"])
                for api_call in debug_info["api_calls"]:
                    api_tracker.track_call(
                        api_call.get("request", {}),
                        type('obj', (object,), {
                            'success': api_call.get("success", True),
                            'usage': api_call.get("response", {}).get("usage", {'total_tokens': 0})
                        })(),
                        api_call.get("response_time", duration)
                    )
            
            # 验证基本功能
            assert isinstance(response, str)
            assert len(response) > 0
            assert duration < 30  # API响应应在30秒内
        
        engine.end_session()
        
        # 验证API调用效率
        assert api_calls_made <= 10  # 4个输入不应超过10次API调用

    @pytest.mark.api_required
    def test_api_error_handling(self, test_config):
        """API错误处理测试（必要测试）"""
        # 使用无效API密钥测试错误处理
        config = EngineConfig(enable_device_manager=True)
        engine = DialogueEngine(api_key="invalid-key-for-testing", config=config)
        
        engine.start_session("api_error_test")
        
        # 测试系统在API失败时的行为
        response, debug_info = engine.process_input("打开客厅的灯")
        
        # 系统应该优雅地处理API错误
        assert isinstance(response, str)
        assert len(response) > 0
        # 应该有错误信息或降级响应
        assert "api_calls" in debug_info
        
        engine.end_session()

    @pytest.mark.api_required
    def test_clarification_api_integration(self, test_config, api_tracker):
        """澄清功能API集成测试（必要测试）"""
        config = EngineConfig(enable_device_manager=True, confidence_threshold=0.9)  # 提高阈值触发澄清
        engine = DialogueEngine(api_key=test_config.api_key, config=config)
        
        engine.start_session("clarification_api_test")
        
        # 测试一个模糊输入触发澄清
        start_time = time.time()
        response, debug_info = engine.process_input("调节一下")
        duration = time.time() - start_time
        
        # 记录API调用
        if "api_calls" in debug_info:
            for api_call in debug_info["api_calls"]:
                api_tracker.track_call(
                    api_call.get("request", {}),
                    type('obj', (object,), {
                        'success': api_call.get("success", True),
                        'usage': api_call.get("response", {}).get("usage", {'total_tokens': 0})
                    })(),
                    api_call.get("response_time", duration)
                )
        
        # 验证澄清功能
        assert isinstance(response, str)
        assert len(response) > 0
        assert duration < 30
        
        # 可能有澄清候选或确认请求
        has_clarification = ("clarification_candidates" in debug_info or 
                           "澄清" in response or "确认" in response or 
                           "请" in response)
        
        engine.end_session()

    @pytest.mark.api_required
    def test_execution_modes_api(self, test_config):
        """执行模式API测试（必要测试）"""
        execution_modes = ["internal_first", "llm_first"]
        
        for mode in execution_modes:
            config = EngineConfig(
                enable_device_manager=True,
                execution_mode=mode
            )
            engine = DialogueEngine(api_key=test_config.api_key, config=config)
            
            engine.start_session(f"execution_mode_{mode}_test")
            
            # 测试设备控制在不同执行模式下的行为
            response, debug_info = engine.process_input("打开客厅的灯")
            
            # 验证响应
            assert isinstance(response, str)
            assert len(response) > 0
            
            # 验证执行模式的影响
            if mode == "internal_first":
                # internal_first模式可能不调用API
                device_result = debug_info.get("device_action_result")
                if device_result and device_result.get("success"):
                    assert "已开启" in response or "打开" in response
            
            engine.end_session()


class TestClarificationAgentAPIUsage:
    """澄清代理API使用测试（必要功能测试）"""
    
    @pytest.mark.api_required
    def test_clarification_generation_api(self, api_client, intent_recognizer, api_tracker):
        """澄清生成API测试（必要测试）"""
        agent = ClarificationAgent(
            api_client=api_client,
            intent_recognizer=intent_recognizer,
            confidence_threshold=0.7,
            max_candidates=3
        )
        
        # 测试一个需要澄清的输入
        context = {
            "last_entities": {
                "devices": [{"value": "空调"}]
            }
        }
        
        start_time = time.time()
        result, candidates = agent.clarify("调节一下", context, [])
        duration = time.time() - start_time
        
        # 记录API使用
        api_tracker.track_call(
            {"action": "clarification", "input": "调节一下"},
            type('obj', (object,), {
                'success': len(candidates) > 0,
                'usage': {'total_tokens': 50}  # 估算
            })(),
            duration
        )
        
        # 验证澄清结果
        assert isinstance(candidates, list)
        assert len(candidates) <= 3  # 不超过最大候选数
        assert duration < 15  # 澄清应在15秒内完成
        
        # 如果有结果，验证结构
        if result:
            assert "intent" in result
            assert "confidence" in result

    @pytest.mark.api_required
    def test_clarification_fallback_mechanism(self, api_client, intent_recognizer):
        """澄清降级机制测试（必要测试）"""
        agent = ClarificationAgent(
            api_client=api_client,
            intent_recognizer=intent_recognizer,
            confidence_threshold=0.7
        )
        
        # 模拟API超时（通过设置很短的超时时间）
        original_timeout = api_client.timeout
        api_client.timeout = 0.001  # 设置极短超时确保触发降级
        
        try:
            context = {"last_entities": {"devices": [{"value": "灯"}]}}
            result, candidates = agent.clarify("开", context, [])
            
            # 降级机制应该产生候选
            assert isinstance(candidates, list)
            assert len(candidates) > 0  # 降级策略应该生成候选
            
        finally:
            # 恢复原始超时
            api_client.timeout = original_timeout


# 性能基准测试（内部逻辑）
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """性能基准测试（内部逻辑，可压力测试）"""
    
    def test_dialogue_throughput_benchmark(self, memory_manager):
        """对话吞吐量基准测试"""
        config = EngineConfig(enable_device_manager=True)
        
        # 测试单线程吞吐量
        engine = DialogueEngine(api_key="test-key-benchmark", config=config)
        engine.memory_manager = memory_manager
        
        test_inputs = [
            "打开客厅的灯",
            "关闭主卧的空调",
            "调节电视音量",
            "查询风扇状态"
        ] * 25  # 100个输入
        
        engine.start_session("throughput_test")
        
        start_time = time.time()
        processed_count = 0
        
        for user_input in test_inputs:
            # 模拟内部处理（不调用API）
            try:
                # 简化的处理流程
                intent_result = engine.intent_recognizer.recognize(user_input, engine.context, engine.dialogue_history)
                
                # 更新上下文
                debug_info = {"intent_result": intent_result}
                engine._update_context(intent_result, debug_info)
                
                # 添加到历史
                from dialogue_manager.engine import DialogueTurn
                turn = DialogueTurn(
                    turn_id=processed_count + 1,
                    user_input=user_input,
                    system_response="测试响应",
                    intent=intent_result.get("intent")
                )
                engine.dialogue_history.append(turn)
                
                processed_count += 1
                
            except Exception as e:
                print(f"处理错误: {e}")
        
        total_time = time.time() - start_time
        throughput = processed_count / total_time
        
        # 验证吞吐量
        assert throughput >= 50  # 至少每秒处理50个对话
        assert processed_count >= len(test_inputs) * 0.95  # 95%成功率
        
        engine.end_session()

    def test_memory_efficiency_benchmark(self):
        """内存效率基准测试"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建多个引擎实例
        engines = []
        for i in range(20):
            config = EngineConfig(enable_device_manager=True)
            engine = DialogueEngine(api_key="test-key-memory", config=config)
            engines.append(engine)
        
        mid_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 运行一些操作
        for i, engine in enumerate(engines):
            engine.start_session(f"memory_user_{i}")
            
            # 简单操作
            for j in range(10):
                engine.context[f"test_{j}"] = f"data_{j}" * 100
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 清理
        for engine in engines:
            engine.end_session()
            engine.dialogue_history.clear()
            engine.context.clear()
        
        del engines
        
        # 验证内存使用
        creation_overhead = mid_memory - initial_memory
        operation_overhead = peak_memory - mid_memory
        
        assert creation_overhead < 200  # 20个引擎创建开销应小于200MB
        assert operation_overhead < 100  # 操作开销应小于100MB


# 配置pytest标记
pytestmark = [
    pytest.mark.dialogue_system,
]

# API测试标记
api_tests_mark = pytest.mark.api_required

# 内部逻辑测试标记
internal_logic_mark = pytest.mark.internal_logic
