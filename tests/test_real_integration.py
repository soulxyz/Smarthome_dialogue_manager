"""真实集成测试

端到端测试完整的智能家居对话系统
"""

import pytest
import time
import json
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Tuple
from pathlib import Path

from dialogue_manager.engine import DialogueEngine, EngineConfig
from dialogue_manager.device_manager import DeviceManager
from dialogue_manager.memory import MemoryManager
from dialogue_manager.api_client import SiliconFlowClient

from tests.fixtures import (
    test_config, api_client, memory_manager, device_manager,
    dialogue_scenarios, api_tracker, performance_test_data
)


class TestCompleteSmartHomeScenarios:
    """完整智能家居场景测试（结合API和内部逻辑）"""
    
    @pytest.mark.integration
    def test_morning_routine_scenario(self, test_config, memory_manager, api_tracker):
        """测试晨起场景的完整对话流程"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True,
            focus_switch_policy="conservative",
            execution_mode="internal_first"
        )
        engine = DialogueEngine(api_key=test_config.api_key, config=config)
        engine.memory_manager = memory_manager
        
        session_id = engine.start_session("morning_routine_user")
        
        # 完整的晨起对话场景
        morning_scenario = [
            # 用户醒来
            ("早上好", "greeting", []),
            ("打开主卧的灯", "device_control", ["device", "location", "action"]),
            ("调节亮度到70%", "device_control", ["device", "number"]),  # 应该使用焦点实体
            
            # 查看状态
            ("现在温度怎么样", "query_status", ["device"]),
            ("把空调温度调到24度", "device_control", ["device", "number"]),
            
            # 移动到客厅
            ("打开客厅的电视", "device_control", ["device", "location", "action"]),
            ("音量调到30", "device_control", ["device", "number"]),  # 焦点切换到电视
            
            # 离开卧室
            ("关掉主卧的灯", "device_control", ["device", "location", "action"]),
            ("空调也关掉", "device_control", ["device", "action"]),  # 省略消解
            
            # 结束
            ("谢谢", "greeting", [])
        ]
        
        dialogue_results = []
        api_calls_count = 0
        
        for i, (user_input, expected_intent, expected_entity_types) in enumerate(morning_scenario):
            start_time = time.time()
            response, debug_info = engine.process_input(user_input)
            duration = time.time() - start_time
            
            # 记录API调用
            if "api_calls" in debug_info:
                api_calls_count += len(debug_info["api_calls"])
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
            assert duration < 30  # 每次交互应在30秒内完成
            
            # 验证意图识别
            actual_intent = debug_info.get("intent_result", {}).get("intent")
            if expected_intent != "greeting":  # 问候可能被识别为其他意图
                assert actual_intent == expected_intent, f"Turn {i+1}: Expected {expected_intent}, got {actual_intent}"
            
            # 验证实体提取
            entities = debug_info.get("intent_result", {}).get("entities", [])
            found_entity_types = {e["entity_type"] for e in entities}
            for expected_type in expected_entity_types:
                assert expected_type in found_entity_types, f"Turn {i+1}: Missing entity type {expected_type}"
            
            dialogue_results.append({
                "turn": i + 1,
                "user_input": user_input,
                "response": response,
                "intent": actual_intent,
                "entities": entities,
                "duration": duration,
                "context_focus": engine.context.get("current_focus", {}).get("value"),
                "api_calls": len(debug_info.get("api_calls", []))
            })
        
        # 验证完整场景
        assert len(dialogue_results) == len(morning_scenario)
        total_duration = sum(r["duration"] for r in dialogue_results)
        assert total_duration < 180  # 整个场景应在3分钟内完成
        
        # 验证API调用效率（对于设备控制为主的场景，API调用应该较少）
        assert api_calls_count <= 20  # 10轮对话不应超过20次API调用
        
        # 验证焦点实体管理
        focus_changes = [r for r in dialogue_results if r["context_focus"]]
        assert len(focus_changes) >= 5  # 应该有多次焦点变化
        
        # 验证设备状态
        final_snapshot = engine.device_manager.snapshot()
        assert final_snapshot["主卧-灯"]["on"] is False  # 主卧灯应该被关闭
        assert final_snapshot["客厅-电视"]["on"] is True   # 客厅电视应该开启
        
        engine.end_session()
        return dialogue_results

    @pytest.mark.integration
    def test_evening_entertainment_scenario(self, test_config, memory_manager):
        """测试晚间娱乐场景"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True,
            focus_switch_policy="aggressive",  # 使用激进切换策略
            execution_mode="llm_first"  # 优先使用LLM
        )
        engine = DialogueEngine(api_key=test_config.api_key, config=config)
        engine.memory_manager = memory_manager
        
        session_id = engine.start_session("evening_entertainment_user")
        
        # 晚间娱乐场景
        evening_scenario = [
            # 准备看电影
            ("我想看电影", "device_control"),
            ("打开客厅的电视", "device_control"),
            ("调到HDMI1", "device_control"),
            ("音量调到50", "device_control"),
            
            # 营造氛围
            ("把客厅的灯调暗一点", "device_control"),
            ("亮度设为30%", "device_control"),
            
            # 舒适度调节
            ("有点热", "device_control"),  # 模糊输入，需要澄清
            ("打开空调", "device_control"),
            ("温度调到22度", "device_control"),
            
            # 结束娱乐
            ("看完了", "greeting"),
            ("关掉电视", "device_control"),
            ("灯光调亮", "device_control"),
            ("谢谢", "greeting")
        ]
        
        scenario_results = []
        
        for user_input, expected_intent in evening_scenario:
            response, debug_info = engine.process_input(user_input)
            
            scenario_results.append({
                "input": user_input,
                "response": response,
                "intent": debug_info.get("intent_result", {}).get("intent"),
                "confidence": debug_info.get("intent_result", {}).get("confidence", 0),
                "focus": engine.context.get("current_focus", {}).get("value"),
                "has_clarification": "clarification_candidates" in debug_info
            })
        
        # 验证场景完成
        assert len(scenario_results) == len(evening_scenario)
        
        # 验证模糊输入处理
        hot_result = next((r for r in scenario_results if r["input"] == "有点热"), None)
        assert hot_result is not None
        # 应该有较低的置信度或澄清候选
        assert hot_result["confidence"] < 0.8 or hot_result["has_clarification"]
        
        # 验证激进焦点切换策略
        focus_values = [r["focus"] for r in scenario_results if r["focus"]]
        unique_focuses = set(focus_values)
        assert len(unique_focuses) >= 3  # 激进策略应产生更多焦点切换
        
        engine.end_session()
        return scenario_results

    @pytest.mark.integration
    def test_multi_room_coordination_scenario(self, test_config, memory_manager):
        """测试多房间协调场景"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True,
            focus_entity_turn_decay=3
        )
        engine = DialogueEngine(api_key=test_config.api_key, config=config)
        engine.memory_manager = memory_manager
        
        session_id = engine.start_session("multi_room_user")
        
        # 多房间协调场景
        multi_room_scenario = [
            # 客厅准备
            ("打开客厅的所有设备", "device_control"),
            ("灯光调到80%", "device_control"),
            ("电视音量调到40", "device_control"),
            
            # 移动到主卧
            ("去主卧了", "query_status"),  # 位置声明
            ("打开主卧的灯", "device_control"),
            ("空调设为睡眠模式", "device_control"),
            ("温度调到26度", "device_control"),
            
            # 远程控制客厅
            ("客厅电视关掉", "device_control"),
            ("客厅灯也关掉", "device_control"),
            
            # 主卧最终设置
            ("把这里的灯调暗", "device_control"),  # "这里"指代当前位置
            ("风扇开一下", "device_control"),
            
            # 检查状态
            ("现在各个房间都怎么样", "query_status")
        ]
        
        room_tracking = []
        
        for user_input, expected_intent in multi_room_scenario:
            response, debug_info = engine.process_input(user_input)
            
            # 跟踪房间相关的实体
            entities = debug_info.get("intent_result", {}).get("entities", [])
            location_entities = [e for e in entities if e["entity_type"] == "location"]
            device_entities = [e for e in entities if e["entity_type"] == "device"]
            
            room_tracking.append({
                "input": user_input,
                "locations": [e["value"] for e in location_entities],
                "devices": [e["value"] for e in device_entities],
                "context_locations": engine.context.get("last_entities", {}).get("locations", []),
                "response": response
            })
        
        # 验证多房间协调
        assert len(room_tracking) == len(multi_room_scenario)
        
        # 验证位置信息传递
        location_mentions = []
        for track in room_tracking:
            location_mentions.extend(track["locations"])
        
        assert "客厅" in location_mentions
        assert "主卧" in location_mentions
        
        # 验证远程控制
        remote_control_inputs = ["客厅电视关掉", "客厅灯也关掉"]
        for remote_input in remote_control_inputs:
            track = next((t for t in room_tracking if t["input"] == remote_input), None)
            assert track is not None
            assert "客厅" in track["locations"] or any("客厅" in str(loc) for loc in track["context_locations"])
        
        # 验证最终设备状态
        final_snapshot = engine.device_manager.snapshot()
        assert final_snapshot["客厅-电视"]["on"] is False  # 应该被远程关闭
        assert final_snapshot["客厅-灯"]["on"] is False    # 应该被远程关闭
        assert final_snapshot["主卧-灯"]["on"] is True     # 应该开启但调暗
        
        engine.end_session()
        return room_tracking


class TestSystemPerformanceIntegration:
    """系统性能集成测试（内部逻辑为主，少量API调用）"""
    
    @pytest.mark.slow
    def test_high_frequency_interactions(self, test_config, memory_manager):
        """测试高频交互性能"""
        config = EngineConfig(
            enable_device_manager=True,
            execution_mode="internal_first"  # 减少API调用
        )
        engine = DialogueEngine(api_key=test_config.api_key, config=config)
        engine.memory_manager = memory_manager
        
        session_id = engine.start_session("high_frequency_user")
        
        # 高频交互测试
        high_freq_inputs = [
            "打开灯", "关闭灯", "调亮", "调暗",
            "开空调", "关空调", "调高温度", "调低温度",
            "开电视", "关电视", "调高音量", "调低音量"
        ] * 50  # 600次交互
        
        start_time = time.time()
        interaction_results = []
        api_call_count = 0
        
        for i, user_input in enumerate(high_freq_inputs):
            interaction_start = time.time()
            response, debug_info = engine.process_input(user_input)
            interaction_duration = time.time() - interaction_start
            
            api_call_count += len(debug_info.get("api_calls", []))
            
            interaction_results.append({
                "turn": i + 1,
                "duration": interaction_duration,
                "success": len(response) > 0
            })
            
            # 每100次交互检查一次性能
            if i > 0 and i % 100 == 0:
                avg_duration = sum(r["duration"] for r in interaction_results[-100:]) / 100
                assert avg_duration < 0.5  # 平均每次交互应在0.5秒内
        
        total_time = time.time() - start_time
        
        # 验证高频性能
        assert total_time < 300  # 600次交互应在5分钟内完成
        throughput = len(high_freq_inputs) / total_time
        assert throughput >= 5  # 每秒至少处理5次交互
        
        # 验证API调用效率
        assert api_call_count <= len(high_freq_inputs) * 0.1  # API调用不超过10%
        
        # 验证成功率
        success_rate = sum(1 for r in interaction_results if r["success"]) / len(interaction_results)
        assert success_rate >= 0.95  # 95%成功率
        
        engine.end_session()
        return {
            "total_interactions": len(high_freq_inputs),
            "total_time": total_time,
            "throughput": throughput,
            "api_calls": api_call_count,
            "success_rate": success_rate
        }

    @pytest.mark.slow
    def test_concurrent_users_simulation(self, test_config):
        """并发用户模拟测试"""
        def simulate_user_session(user_id, interaction_count=20):
            """模拟单个用户会话"""
            config = EngineConfig(
                enable_device_manager=True,
                execution_mode="internal_first"
            )
            engine = DialogueEngine(api_key=test_config.api_key, config=config)
            
            session_id = engine.start_session(f"concurrent_user_{user_id}")
            
            user_interactions = [
                f"打开用户{user_id}的灯",
                f"调节亮度",
                f"查询状态",
                f"关闭设备"
            ] * (interaction_count // 4)
            
            session_results = []
            for interaction in user_interactions:
                try:
                    start = time.time()
                    response, debug_info = engine.process_input(interaction)
                    duration = time.time() - start
                    
                    session_results.append({
                        "user_id": user_id,
                        "interaction": interaction,
                        "duration": duration,
                        "success": len(response) > 0,
                        "api_calls": len(debug_info.get("api_calls", []))
                    })
                except Exception as e:
                    session_results.append({
                        "user_id": user_id,
                        "interaction": interaction,
                        "error": str(e),
                        "success": False
                    })
            
            engine.end_session()
            return session_results
        
        # 并发用户测试
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(simulate_user_session, i, 20) for i in range(20)]
            all_user_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # 汇总结果
        total_interactions = sum(len(results) for results in all_user_results)
        successful_interactions = sum(
            sum(1 for r in results if r["success"]) for results in all_user_results
        )
        total_api_calls = sum(
            sum(r.get("api_calls", 0) for r in results) for results in all_user_results
        )
        
        # 验证并发性能
        assert total_time < 120  # 20个并发用户应在2分钟内完成
        assert len(all_user_results) == 20  # 所有用户会话都完成
        
        success_rate = successful_interactions / total_interactions
        assert success_rate >= 0.9  # 90%交互成功
        
        # 验证资源使用效率
        api_efficiency = total_api_calls / total_interactions
        assert api_efficiency <= 0.2  # API调用率不超过20%
        
        return {
            "concurrent_users": 20,
            "total_interactions": total_interactions,
            "success_rate": success_rate,
            "total_time": total_time,
            "api_efficiency": api_efficiency
        }

    @pytest.mark.slow
    def test_long_running_session_stability(self, test_config, memory_manager):
        """长时间运行会话稳定性测试"""
        config = EngineConfig(
            enable_device_manager=True,
            max_history_length=100,  # 限制历史长度
            execution_mode="internal_first"
        )
        engine = DialogueEngine(api_key=test_config.api_key, config=config)
        engine.memory_manager = memory_manager
        
        session_id = engine.start_session("long_running_user")
        
        # 模拟长时间会话（1000次交互）
        base_interactions = [
            "打开客厅的灯", "调节亮度到{}%", "查询灯状态", "关闭灯",
            "打开空调", "设置温度{}度", "查询空调状态", "关闭空调",
            "打开电视", "音量调到{}", "查询电视状态", "关闭电视"
        ]
        
        long_session_results = []
        memory_snapshots = []
        
        start_time = time.time()
        
        for i in range(1000):
            # 选择交互模式
            base_interaction = base_interactions[i % len(base_interactions)]
            
            # 填充参数
            if "{}" in base_interaction:
                if "亮度" in base_interaction:
                    user_input = base_interaction.format(20 + (i % 8) * 10)
                elif "温度" in base_interaction:
                    user_input = base_interaction.format(18 + (i % 12))
                elif "音量" in base_interaction:
                    user_input = base_interaction.format(10 + (i % 9) * 10)
                else:
                    user_input = base_interaction.format(i % 100)
            else:
                user_input = base_interaction
            
            try:
                interaction_start = time.time()
                response, debug_info = engine.process_input(user_input)
                interaction_duration = time.time() - interaction_start
                
                long_session_results.append({
                    "turn": i + 1,
                    "duration": interaction_duration,
                    "context_size": len(str(engine.context)),
                    "history_length": len(engine.dialogue_history),
                    "success": len(response) > 0
                })
                
                # 每100次记录内存快照
                if i % 100 == 0:
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    memory_snapshots.append({
                        "turn": i + 1,
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                        "context_keys": len(engine.context),
                        "history_length": len(engine.dialogue_history)
                    })
                
            except Exception as e:
                long_session_results.append({
                    "turn": i + 1,
                    "error": str(e),
                    "success": False
                })
        
        total_session_time = time.time() - start_time
        
        # 验证长时间运行稳定性
        assert total_session_time < 600  # 1000次交互应在10分钟内完成
        
        # 验证成功率保持稳定
        success_rate = sum(1 for r in long_session_results if r["success"]) / len(long_session_results)
        assert success_rate >= 0.95  # 长时间运行仍保持95%成功率
        
        # 验证性能不显著下降
        early_results = long_session_results[:100]
        late_results = long_session_results[-100:]
        
        early_avg_duration = sum(r["duration"] for r in early_results if r["success"]) / len([r for r in early_results if r["success"]])
        late_avg_duration = sum(r["duration"] for r in late_results if r["success"]) / len([r for r in late_results if r["success"]])
        
        performance_degradation = (late_avg_duration - early_avg_duration) / early_avg_duration
        assert performance_degradation < 0.5  # 性能下降不超过50%
        
        # 验证内存使用稳定
        if len(memory_snapshots) >= 2:
            initial_memory = memory_snapshots[0]["memory_mb"]
            final_memory = memory_snapshots[-1]["memory_mb"]
            memory_growth = (final_memory - initial_memory) / initial_memory
            assert memory_growth < 2.0  # 内存增长不超过200%
        
        # 验证历史长度控制
        final_history_length = long_session_results[-1]["history_length"]
        assert final_history_length <= config.max_history_length
        
        engine.end_session()
        
        return {
            "total_turns": 1000,
            "total_time": total_session_time,
            "success_rate": success_rate,
            "performance_degradation": performance_degradation,
            "memory_snapshots": memory_snapshots,
            "final_history_length": final_history_length
        }


class TestErrorRecoveryIntegration:
    """错误恢复集成测试（混合API和内部逻辑）"""
    
    @pytest.mark.integration
    def test_api_failure_graceful_degradation(self, test_config, memory_manager):
        """测试API失败时的优雅降级"""
        config = EngineConfig(
            enable_device_manager=True,
            execution_mode="llm_first"  # 优先使用LLM，测试降级
        )
        engine = DialogueEngine(api_key=test_config.api_key, config=config)
        engine.memory_manager = memory_manager
        
        session_id = engine.start_session("api_failure_test")
        
        # 正常操作
        response1, debug1 = engine.process_input("打开客厅的灯")
        assert isinstance(response1, str) and len(response1) > 0
        
        # 模拟API失败（修改API密钥）
        original_api_key = engine.api_client.api_key
        engine.api_client.api_key = "invalid_key_for_testing"
        
        # API失败时的操作
        degradation_responses = []
        degradation_inputs = [
            "调节亮度到50%",  # 设备控制，应该能降级到内部处理
            "查询灯状态",      # 查询操作，应该能降级到内部处理
            "关闭灯",          # 设备控制，应该能降级到内部处理
            "谢谢",           # 问候，可能需要API但应该有降级
        ]
        
        for user_input in degradation_inputs:
            response, debug_info = engine.process_input(user_input)
            degradation_responses.append({
                "input": user_input,
                "response": response,
                "has_response": len(response) > 0,
                "api_errors": any(
                    not call.get("success", True) 
                    for call in debug_info.get("api_calls", [])
                ),
                "device_result": debug_info.get("device_action_result", {}).get("success")
            })
        
        # 恢复API密钥
        engine.api_client.api_key = original_api_key
        
        # 恢复后的操作
        response_after_recovery, _ = engine.process_input("打开电视")
        
        # 验证降级行为
        assert len(degradation_responses) == len(degradation_inputs)
        
        # 所有操作都应该有响应
        for result in degradation_responses:
            assert result["has_response"], f"No response for: {result['input']}"
        
        # 设备控制操作应该成功（降级到内部处理）
        device_operations = [r for r in degradation_responses if r["device_result"] is not None]
        if device_operations:
            device_success_rate = sum(1 for r in device_operations if r["device_result"]) / len(device_operations)
            assert device_success_rate >= 0.8  # 80%设备操作成功
        
        # 恢复后应该正常工作
        assert isinstance(response_after_recovery, str) and len(response_after_recovery) > 0
        
        engine.end_session()
        return degradation_responses

    @pytest.mark.integration  
    def test_partial_system_failure_recovery(self, test_config, memory_manager):
        """测试部分系统故障恢复"""
        config = EngineConfig(enable_device_manager=True)
        engine = DialogueEngine(api_key=test_config.api_key, config=config)
        engine.memory_manager = memory_manager
        
        session_id = engine.start_session("partial_failure_test")
        
        # 正常建立上下文
        response1, _ = engine.process_input("打开主卧的空调")
        
        # 模拟设备管理器部分故障（移除某些设备）
        original_devices = engine.device_manager.devices.copy()
        original_index = engine.device_manager._index.copy()
        
        # 移除空调设备模拟故障
        engine.device_manager.devices = [d for d in engine.device_manager.devices if d.device_type != "空调"]
        engine.device_manager._index = {k: v for k, v in engine.device_manager._index.items() if k[0] != "空调"}
        
        # 故障状态下的操作
        failure_responses = []
        failure_inputs = [
            "调高温度",        # 应该失败，但有优雅错误
            "打开客厅的灯",    # 应该成功，灯设备仍存在
            "关闭空调",        # 应该失败，但有优雅错误
            "查询所有设备状态"  # 应该成功，返回可用设备
        ]
        
        for user_input in failure_inputs:
            response, debug_info = engine.process_input(user_input)
            failure_responses.append({
                "input": user_input,
                "response": response,
                "success": len(response) > 0,
                "device_action": debug_info.get("device_action_result"),
                "device_query": debug_info.get("device_query_result")
            })
        
        # 恢复设备管理器
        engine.device_manager.devices = original_devices
        engine.device_manager._index = original_index
        
        # 恢复后测试
        recovery_response, _ = engine.process_input("现在空调可以用了吗")
        
        # 验证故障处理
        assert len(failure_responses) == len(failure_inputs)
        
        # 所有操作都应该有响应
        for result in failure_responses:
            assert result["success"], f"No response for: {result['input']}"
        
        # 涉及故障设备的操作应该有适当的错误消息
        ac_operations = [r for r in failure_responses if "空调" in r["input"]]
        for ac_op in ac_operations:
            device_result = ac_op.get("device_action") or ac_op.get("device_query")
            if device_result:
                assert not device_result.get("success", True) or "未找到" in device_result.get("message", "")
        
        # 不涉及故障设备的操作应该成功
        light_operation = next((r for r in failure_responses if "灯" in r["input"]), None)
        if light_operation:
            device_result = light_operation.get("device_action")
            if device_result:
                assert device_result.get("success", False)
        
        # 恢复后应该正常
        assert isinstance(recovery_response, str) and len(recovery_response) > 0
        
        engine.end_session()
        return failure_responses


@pytest.mark.integration
class TestCompleteSystemValidation:
    """完整系统验证测试"""
    
    def test_end_to_end_system_validation(self, test_config, api_tracker):
        """端到端系统验证测试"""
        # 创建完整系统
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True,
            focus_switch_policy="conservative",
            focus_entity_turn_decay=2,
            execution_mode="internal_first",
            max_history_length=50
        )
        
        with MemoryManager(db_path=":memory:") as memory_manager:
            engine = DialogueEngine(api_key=test_config.api_key, config=config)
            engine.memory_manager = memory_manager
            
            # 完整的系统验证场景
            validation_scenario = [
                # 系统初始化和用户识别
                ("你好，我是新用户", "greeting"),
                
                # 基础设备控制
                ("打开客厅的灯", "device_control"),
                ("调节亮度到75%", "device_control"),
                
                # 焦点实体和省略消解
                ("关掉它", "device_control"),  # 代词消解
                ("再开一下", "device_control"),  # 省略消解
                
                # 多设备控制
                ("打开空调", "device_control"),
                ("温度设为24度", "device_control"),
                
                # 焦点切换
                ("电视声音太大", "device_control"),  # 模糊表达
                ("调小一点", "device_control"),  # 焦点切换到电视
                
                # 状态查询
                ("现在客厅都有什么设备开着", "query_status"),
                
                # 场景控制
                ("我要睡觉了", "device_control"),
                ("关掉所有灯", "device_control"),
                ("空调调到睡眠模式", "device_control"),
                
                # 系统反馈
                ("谢谢你的帮助", "greeting"),
            ]
            
            session_id = engine.start_session("system_validation_user")
            
            validation_results = {
                "dialogue_turns": [],
                "focus_changes": [],
                "device_states": [],
                "api_calls": 0,
                "errors": [],
                "performance_metrics": {
                    "total_time": 0,
                    "avg_response_time": 0,
                    "success_rate": 0
                }
            }
            
            start_time = time.time()
            
            for i, (user_input, expected_intent) in enumerate(validation_scenario):
                turn_start = time.time()
                
                try:
                    response, debug_info = engine.process_input(user_input)
                    turn_duration = time.time() - turn_start
                    
                    # 记录对话轮次
                    turn_result = {
                        "turn": i + 1,
                        "user_input": user_input,
                        "response": response,
                        "expected_intent": expected_intent,
                        "actual_intent": debug_info.get("intent_result", {}).get("intent"),
                        "confidence": debug_info.get("intent_result", {}).get("confidence", 0),
                        "entities": debug_info.get("intent_result", {}).get("entities", []),
                        "duration": turn_duration,
                        "success": len(response) > 0
                    }
                    validation_results["dialogue_turns"].append(turn_result)
                    
                    # 记录焦点变化
                    if "current_focus" in engine.context:
                        validation_results["focus_changes"].append({
                            "turn": i + 1,
                            "focus": engine.context["current_focus"]["value"],
                            "turn_count": engine.context["current_focus"]["turn_count"]
                        })
                    
                    # 记录设备状态
                    validation_results["device_states"].append({
                        "turn": i + 1,
                        "snapshot": engine.device_manager.snapshot()
                    })
                    
                    # 记录API调用
                    api_calls = len(debug_info.get("api_calls", []))
                    validation_results["api_calls"] += api_calls
                    
                    for api_call in debug_info.get("api_calls", []):
                        api_tracker.track_call(
                            api_call.get("request", {}),
                            type('obj', (object,), {
                                'success': api_call.get("success", True),
                                'usage': api_call.get("response", {}).get("usage", {'total_tokens': 0})
                            })(),
                            api_call.get("response_time", turn_duration)
                        )
                    
                except Exception as e:
                    validation_results["errors"].append({
                        "turn": i + 1,
                        "input": user_input,
                        "error": str(e)
                    })
            
            total_time = time.time() - start_time
            
            # 计算性能指标
            successful_turns = [t for t in validation_results["dialogue_turns"] if t["success"]]
            validation_results["performance_metrics"] = {
                "total_time": total_time,
                "avg_response_time": sum(t["duration"] for t in successful_turns) / len(successful_turns) if successful_turns else 0,
                "success_rate": len(successful_turns) / len(validation_scenario),
                "total_turns": len(validation_scenario),
                "api_efficiency": validation_results["api_calls"] / len(validation_scenario)
            }
            
            engine.end_session()
            
            # 验证系统完整性
            assert validation_results["performance_metrics"]["success_rate"] >= 0.9  # 90%成功率
            assert validation_results["performance_metrics"]["avg_response_time"] < 5.0  # 平均5秒内响应
            assert len(validation_results["errors"]) <= 2  # 最多2个错误
            assert validation_results["performance_metrics"]["api_efficiency"] <= 1.0  # API调用效率
            
            # 验证焦点管理
            assert len(validation_results["focus_changes"]) >= 3  # 至少3次焦点变化
            
            # 验证设备状态一致性
            final_snapshot = validation_results["device_states"][-1]["snapshot"]
            assert isinstance(final_snapshot, dict)
            assert len(final_snapshot) > 0
            
            return validation_results


# 配置pytest标记
pytestmark = [
    pytest.mark.integration,
    pytest.mark.system_test
]
