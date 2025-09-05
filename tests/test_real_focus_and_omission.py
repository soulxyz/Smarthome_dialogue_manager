"""真实焦点实体和省略消解测试

专注于测试焦点实体管理和省略消解的真实场景
"""

import pytest
import time
import concurrent.futures
from typing import Dict, List, Any

from dialogue_manager.engine import DialogueEngine, EngineConfig
from dialogue_manager.intent import IntentRecognizer

from tests.fixtures import (
    test_config, device_manager, memory_manager, 
    dialogue_engine, intent_recognizer,
    dialogue_scenarios, api_tracker
)


class TestFocusEntityManagementInternal:
    """焦点实体管理内部逻辑测试（可压力测试）"""
    
    def test_focus_creation_and_updates_stress(self, memory_manager):
        """压力测试：焦点实体创建和更新"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True,
            focus_entity_turn_decay=3
        )
        engine = DialogueEngine(api_key="test-key-focus", config=config)
        engine.memory_manager = memory_manager
        
        engine.start_session("focus_stress_test")
        
        # 大量焦点创建和切换
        devices = ["灯", "空调", "电视", "风扇", "音响", "窗帘", "门锁", "热水器"]
        rooms = ["客厅", "主卧", "次卧", "厨房", "书房", "阳台", "卫生间"]
        
        start_time = time.time()
        focus_changes = []
        
        for i in range(500):
            device = devices[i % len(devices)]
            room = rooms[i % len(rooms)]
            
            # 模拟设备控制意图
            intent_result = {
                "intent": "device_control",
                "confidence": 0.9,
                "entities": [
                    {
                        "entity_type": "device",
                        "value": device,
                        "start_pos": 0,
                        "end_pos": len(device)
                    },
                    {
                        "entity_type": "location", 
                        "value": room,
                        "start_pos": len(device) + 1,
                        "end_pos": len(device) + 1 + len(room)
                    }
                ],
                "original_text": f"打开{room}的{device}"
            }
            
            # 更新上下文和焦点
            debug_info = {"intent_result": intent_result}
            engine._update_context(intent_result, debug_info, i + 1)
            
            # 记录焦点变化
            if "current_focus" in engine.context:
                focus_info = engine.context["current_focus"].copy()
                focus_info["turn"] = i + 1
                focus_changes.append(focus_info)
        
        focus_time = time.time() - start_time
        
        # 验证焦点管理性能
        assert focus_time < 2.0  # 500次焦点操作应在2秒内完成
        assert len(focus_changes) >= 400  # 大部分操作都应创建焦点
        
        # 验证焦点切换逻辑
        unique_devices = set(fc["value"] for fc in focus_changes)
        assert len(unique_devices) == len(devices)  # 所有设备都被设为焦点
        
        engine.end_session()

    def test_focus_decay_mechanism_stress(self, memory_manager):
        """压力测试：焦点实体衰减机制"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True,
            focus_entity_turn_decay=2
        )
        engine = DialogueEngine(api_key="test-key-decay", config=config)
        engine.memory_manager = memory_manager
        
        engine.start_session("decay_stress_test")
        
        # 建立初始焦点
        initial_intent = {
            "intent": "device_control",
            "confidence": 0.9,
            "entities": [{"entity_type": "device", "value": "灯", "start_pos": 0, "end_pos": 1}],
            "original_text": "打开灯"
        }
        debug_info = {"intent_result": initial_intent}
        engine._update_context(initial_intent, debug_info, 1)
        
        assert "current_focus" in engine.context
        assert engine.context["current_focus"]["value"] == "灯"
        
        # 大量非设备相关的对话
        start_time = time.time()
        decay_steps = []
        
        for i in range(100):
            # 非设备控制意图
            other_intent = {
                "intent": "greeting" if i % 2 == 0 else "query_weather",
                "confidence": 0.8,
                "entities": [],
                "original_text": "你好" if i % 2 == 0 else "天气怎么样"
            }
            
            debug_info = {"intent_result": other_intent}
            engine._update_context(other_intent, debug_info, i + 2)
            
            # 记录焦点状态
            if "current_focus" in engine.context:
                focus_info = engine.context["current_focus"].copy()
                focus_info["turn"] = i + 2
                decay_steps.append(focus_info)
            else:
                decay_steps.append({"status": "expired", "turn": i + 2})
        
        decay_time = time.time() - start_time
        
        # 验证衰减性能
        assert decay_time < 1.0  # 100次衰减检查应在1秒内完成
        
        # 验证衰减逻辑
        expired_step = None
        for step in decay_steps:
            if step.get("status") == "expired":
                expired_step = step
                break
        
        assert expired_step is not None  # 焦点应该在某个时刻过期
        assert expired_step["turn"] <= 5  # 应该在第5轮或之前过期（初始+2轮衰减）
        
        engine.end_session()

    def test_focus_switching_strategies_stress(self, memory_manager):
        """压力测试：焦点切换策略"""
        strategies = ["conservative", "aggressive"]
        
        for strategy in strategies:
            config = EngineConfig(
                enable_device_manager=True,
                enable_context_entity_fill=True,
                focus_switch_policy=strategy,
                focus_entity_turn_decay=3
            )
            engine = DialogueEngine(api_key="test-key-strategy", config=config)
            engine.memory_manager = memory_manager
            
            engine.start_session(f"strategy_{strategy}_test")
            
            # 建立初始焦点
            initial_intent = {
                "intent": "device_control",
                "confidence": 0.9,
                "entities": [{"entity_type": "device", "value": "灯", "start_pos": 0, "end_pos": 1}],
                "original_text": "打开灯"
            }
            debug_info = {"intent_result": initial_intent}
            engine._update_context(initial_intent, debug_info, 1)
            
            # 测试不同意图对焦点的影响
            test_cases = [
                ("query_status", "空调状态怎么样", "空调"),
                ("device_control", "打开电视", "电视"),
                ("query_status", "风扇转得快吗", "风扇"),
                ("device_control", "关闭音响", "音响")
            ] * 50  # 200个测试案例
            
            start_time = time.time()
            switch_results = []
            
            for i, (intent_type, text, device) in enumerate(test_cases):
                test_intent = {
                    "intent": intent_type,
                    "confidence": 0.8,
                    "entities": [{"entity_type": "device", "value": device, "start_pos": 0, "end_pos": len(device)}],
                    "original_text": text
                }
                
                old_focus = engine.context.get("current_focus", {}).get("value")
                
                debug_info = {"intent_result": test_intent}
                engine._update_context(test_intent, debug_info, i + 2)
                
                new_focus = engine.context.get("current_focus", {}).get("value")
                
                switch_results.append({
                    "strategy": strategy,
                    "intent": intent_type,
                    "old_focus": old_focus,
                    "new_focus": new_focus,
                    "switched": old_focus != new_focus
                })
            
            strategy_time = time.time() - start_time
            
            # 验证策略性能
            assert strategy_time < 1.0  # 200次策略判断应在1秒内完成
            
            # 验证策略行为
            if strategy == "conservative":
                # 保守策略：查询不应切换焦点
                query_switches = [r for r in switch_results if r["intent"] == "query_status" and r["switched"]]
                control_switches = [r for r in switch_results if r["intent"] == "device_control" and r["switched"]]
                
                # 保守策略下，查询切换应该少于控制切换
                assert len(query_switches) <= len(control_switches)
            
            elif strategy == "aggressive":
                # 激进策略：任何设备提及都可能切换焦点
                total_switches = [r for r in switch_results if r["switched"]]
                assert len(total_switches) >= len(switch_results) * 0.6  # 至少60%发生切换
            
            engine.end_session()

    def test_concurrent_focus_management(self, memory_manager):
        """测试并发焦点管理"""
        def manage_focus_session(session_id):
            """管理单个会话的焦点"""
            config = EngineConfig(
                enable_device_manager=True,
                enable_context_entity_fill=True,
                focus_entity_turn_decay=2
            )
            engine = DialogueEngine(api_key="test-key-concurrent", config=config)
            engine.memory_manager = memory_manager
            
            engine.start_session(f"concurrent_focus_{session_id}")
            
            devices = ["灯", "空调", "电视", "风扇"]
            focus_changes = []
            
            for i in range(20):
                device = devices[i % len(devices)]
                intent_result = {
                    "intent": "device_control",
                    "confidence": 0.9,
                    "entities": [{"entity_type": "device", "value": device, "start_pos": 0, "end_pos": len(device)}],
                    "original_text": f"打开{device}"
                }
                
                debug_info = {"intent_result": intent_result}
                engine._update_context(intent_result, debug_info, i + 1)
                
                if "current_focus" in engine.context:
                    focus_changes.append(engine.context["current_focus"]["value"])
            
            engine.end_session()
            return focus_changes
        
        # 并发运行多个会话
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(manage_focus_session, i) for i in range(10)]
            all_focus_changes = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        concurrent_time = time.time() - start_time
        
        # 验证并发性能
        assert concurrent_time < 3.0  # 10个并发会话应在3秒内完成
        assert len(all_focus_changes) == 10
        
        # 验证焦点管理的独立性
        for focus_changes in all_focus_changes:
            assert len(focus_changes) >= 10  # 每个会话至少有10次焦点变化
            unique_devices = set(focus_changes)
            assert len(unique_devices) <= 4  # 不超过设备数量


class TestOmissionResolutionInternal:
    """省略消解内部逻辑测试（可压力测试）"""
    
    def test_omission_resolution_patterns_stress(self, device_manager):
        """压力测试：省略消解模式"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True,
            focus_entity_turn_decay=3
        )
        recognizer = IntentRecognizer(config=config, device_manager=device_manager)
        
        # 建立各种上下文场景
        contexts = []
        for i in range(200):
            context = {
                "current_focus": {
                    "value": f"设备_{i % 10}",
                    "turn_count": i % 3,
                    "confidence": 0.8 + (i % 2) * 0.1
                },
                "last_entities": {
                    "devices": [{"value": f"设备_{j}"} for j in range(i % 5)],
                    "actions": [{"value": f"动作_{j}"} for j in range(i % 3)]
                }
            }
            contexts.append(context)
        
        # 省略主语的测试输入
        omission_inputs = [
            "关掉", "打开", "调节一下", "设置", "调高", "调低",
            "关闭", "启动", "停止", "增加", "减少", "切换"
        ]
        
        start_time = time.time()
        resolution_results = []
        
        for i, context in enumerate(contexts):
            user_input = omission_inputs[i % len(omission_inputs)]
            
            # 执行省略消解
            result = recognizer._extract_entities_with_context(
                user_input, "device_control", context, []
            )
            
            # 检查是否成功补充设备实体
            device_entities = [e for e in result if e.entity_type == "device"]
            action_entities = [e for e in result if e.entity_type == "action"]
            
            resolution_results.append({
                "input": user_input,
                "context_device": context["current_focus"]["value"],
                "resolved_devices": [e.value for e in device_entities],
                "resolved_actions": [e.value for e in action_entities],
                "successful": len(device_entities) > 0
            })
        
        resolution_time = time.time() - start_time
        
        # 验证省略消解性能
        assert resolution_time < 2.0  # 200次省略消解应在2秒内完成
        
        # 验证消解成功率
        successful_resolutions = [r for r in resolution_results if r["successful"]]
        success_rate = len(successful_resolutions) / len(resolution_results)
        assert success_rate >= 0.8  # 至少80%的省略应该被成功消解
        
        # 验证消解准确性
        for result in successful_resolutions[:10]:  # 检查前10个成功案例
            assert result["context_device"] in result["resolved_devices"]

    def test_pronoun_resolution_stress(self, device_manager):
        """压力测试：代词消解"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True
        )
        recognizer = IntentRecognizer(config=config, device_manager=device_manager)
        
        # 代词测试
        pronouns = ["它", "这个", "那个", "他", "她"]
        pronoun_inputs = []
        
        for i in range(500):
            pronoun = pronouns[i % len(pronouns)]
            action = ["打开", "关闭", "调节", "设置"][i % 4]
            pronoun_inputs.append(f"{action}{pronoun}")
        
        # 创建上下文
        contexts = []
        for i in range(500):
            context = {
                "current_focus": {
                    "value": f"设备_{i % 20}",
                    "turn_count": 0,
                    "confidence": 0.9
                }
            }
            contexts.append(context)
        
        start_time = time.time()
        pronoun_resolutions = []
        
        for i, (user_input, context) in enumerate(zip(pronoun_inputs, contexts)):
            # 执行代词消解
            result = recognizer.recognize(user_input, context, [])
            
            # 检查代词是否被正确解析
            entities = result.get("entities", [])
            device_entities = [e for e in entities if e["entity_type"] == "device"]
            
            pronoun_resolutions.append({
                "input": user_input,
                "expected_device": context["current_focus"]["value"],
                "resolved_devices": [e["value"] for e in device_entities],
                "successful": len(device_entities) > 0 and context["current_focus"]["value"] in [e["value"] for e in device_entities]
            })
        
        pronoun_time = time.time() - start_time
        
        # 验证代词消解性能
        assert pronoun_time < 3.0  # 500次代词消解应在3秒内完成
        
        # 验证代词消解成功率
        successful_pronouns = [r for r in pronoun_resolutions if r["successful"]]
        pronoun_success_rate = len(successful_pronouns) / len(pronoun_resolutions)
        assert pronoun_success_rate >= 0.7  # 至少70%的代词应该被正确消解

    def test_complex_omission_scenarios_stress(self, device_manager):
        """压力测试：复杂省略场景"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True,
            focus_entity_turn_decay=3
        )
        recognizer = IntentRecognizer(config=config, device_manager=device_manager)
        
        # 创建复杂的省略场景
        complex_scenarios = []
        for i in range(100):
            # 多层省略：既省略设备又省略位置
            scenario = {
                "context": {
                    "current_focus": {"value": f"设备_{i % 10}", "turn_count": 0},
                    "last_entities": {
                        "locations": [{"value": f"房间_{i % 5}"}],
                        "devices": [{"value": f"设备_{j}"} for j in range(i % 3)],
                        "actions": [{"value": "打开"}]
                    }
                },
                "history": [
                    {
                        "user_input": f"打开房间_{i % 5}的设备_{i % 10}",
                        "entities": [
                            {"type": "device", "value": f"设备_{i % 10}"},
                            {"type": "location", "value": f"房间_{i % 5}"}
                        ]
                    }
                ],
                "input": ["关掉", "调亮一点", "温度高一些", "声音小一点"][i % 4]
            }
            complex_scenarios.append(scenario)
        
        start_time = time.time()
        complex_results = []
        
        for scenario in complex_scenarios:
            result = recognizer._extract_entities_with_context(
                scenario["input"], 
                "device_control", 
                scenario["context"], 
                scenario["history"]
            )
            
            device_entities = [e for e in result if e.entity_type == "device"]
            location_entities = [e for e in result if e.entity_type == "location"]
            
            complex_results.append({
                "input": scenario["input"],
                "expected_device": scenario["context"]["current_focus"]["value"],
                "expected_location": scenario["context"]["last_entities"]["locations"][0]["value"],
                "found_devices": [e.value for e in device_entities],
                "found_locations": [e.value for e in location_entities],
                "device_resolved": len(device_entities) > 0,
                "location_inherited": len(location_entities) > 0
            })
        
        complex_time = time.time() - start_time
        
        # 验证复杂省略性能
        assert complex_time < 2.0  # 100次复杂省略应在2秒内完成
        
        # 验证复杂省略效果
        device_resolution_rate = sum(1 for r in complex_results if r["device_resolved"]) / len(complex_results)
        location_inheritance_rate = sum(1 for r in complex_results if r["location_inherited"]) / len(complex_results)
        
        assert device_resolution_rate >= 0.8  # 80%的设备省略被解决
        assert location_inheritance_rate >= 0.6  # 60%的位置被继承

    def test_omission_with_disabled_context_fill(self, device_manager):
        """测试禁用上下文填充时的行为"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=False  # 禁用上下文实体填充
        )
        recognizer = IntentRecognizer(config=config, device_manager=device_manager)
        
        # 有焦点实体的上下文
        context_with_focus = {
            "current_focus": {
                "value": "空调",
                "turn_count": 0,
                "confidence": 0.9
            }
        }
        
        # 省略主语的输入
        omission_inputs = ["关掉", "打开", "调节", "设置温度", "调高风速"]
        
        start_time = time.time()
        disabled_results = []
        
        for user_input in omission_inputs:
            result = recognizer._extract_entities_with_context(
                user_input, "device_control", context_with_focus, []
            )
            
            device_entities = [e for e in result if e.entity_type == "device"]
            disabled_results.append({
                "input": user_input,
                "devices_found": len(device_entities),
                "entity_filled": any(e.value == "空调" for e in device_entities)
            })
        
        disabled_time = time.time() - start_time
        
        # 验证禁用状态下的行为
        assert disabled_time < 0.1  # 禁用状态下应该很快
        
        # 禁用上下文填充时，不应该从焦点补充实体
        filled_count = sum(1 for r in disabled_results if r["entity_filled"])
        assert filled_count == 0  # 不应该有任何实体被填充


class TestFocusAndOmissionAPIIntegration:
    """焦点和省略消解API集成测试（必要功能测试）"""
    
    @pytest.mark.api_required
    def test_focus_tracking_with_real_api(self, test_config, memory_manager, api_tracker):
        """使用真实API测试焦点跟踪"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True,
            focus_entity_turn_decay=2
        )
        engine = DialogueEngine(api_key=test_config.api_key, config=config)
        engine.memory_manager = memory_manager
        
        engine.start_session("focus_api_test")
        
        # 核心焦点场景测试（节约API调用）
        focus_scenario = [
            ("打开客厅的灯", "灯"),      # 建立焦点
            ("调节亮度到80%", "灯"),     # 使用焦点
            ("关掉它", "灯"),           # 省略消解
        ]
        
        for i, (user_input, expected_focus) in enumerate(focus_scenario):
            start_time = time.time()
            response, debug_info = engine.process_input(user_input)
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
            
            # 验证响应
            assert isinstance(response, str)
            assert len(response) > 0
            assert duration < 30
            
            # 验证焦点实体
            if "current_focus" in engine.context:
                assert engine.context["current_focus"]["value"] == expected_focus
        
        engine.end_session()

    @pytest.mark.api_required
    def test_omission_resolution_with_real_api(self, test_config, memory_manager):
        """使用真实API测试省略消解"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True
        )
        engine = DialogueEngine(api_key=test_config.api_key, config=config)
        engine.memory_manager = memory_manager
        
        engine.start_session("omission_api_test")
        
        # 省略消解场景（最小化API调用）
        response1, _ = engine.process_input("打开主卧的空调")  # 建立上下文
        response2, debug_info = engine.process_input("关掉")  # 测试省略消解
        
        # 验证省略消解
        assert isinstance(response2, str)
        assert len(response2) > 0
        
        # 验证实体被正确补充
        entities = debug_info.get("intent_result", {}).get("entities", [])
        device_entities = [e for e in entities if e["entity_type"] == "device"]
        
        # 应该从上下文补充设备实体
        assert len(device_entities) > 0
        assert any(e["value"] == "空调" for e in device_entities)
        
        engine.end_session()

    @pytest.mark.api_required
    def test_focus_switch_policies_with_api(self, test_config, memory_manager):
        """使用真实API测试焦点切换策略"""
        for policy in ["conservative"]:  # 只测试一种策略以节约API
            config = EngineConfig(
                enable_device_manager=True,
                focus_switch_policy=policy
            )
            engine = DialogueEngine(api_key=test_config.api_key, config=config)
            engine.memory_manager = memory_manager
            
            engine.start_session(f"policy_{policy}_api_test")
            
            # 最小化API调用的策略测试
            engine.process_input("打开客厅的灯")          # 建立焦点
            engine.process_input("空调状态怎么样")        # 查询意图
            response, debug_info = engine.process_input("打开空调")  # 控制意图
            
            # 验证最终状态
            assert isinstance(response, str)
            assert len(response) > 0
            
            # 验证焦点切换（保守策略下，设备控制应该切换焦点）
            if "current_focus" in engine.context:
                final_focus = engine.context["current_focus"]["value"]
                assert final_focus == "空调"  # 焦点应该切换到空调
            
            engine.end_session()


# 配置pytest标记
pytestmark = [
    pytest.mark.focus_management,
    pytest.mark.omission_resolution
]
