"""真实设备管理器测试

分离API测试和内部逻辑测试，内部逻辑可以进行压力测试
"""

import pytest
import time
import concurrent.futures
import threading
from typing import Dict, List, Any
from unittest.mock import patch

from dialogue_manager.device_manager import (
    DeviceManager, LightDevice, AirConditionerDevice, 
    TVDevice, FanDevice, Device
)

from tests.fixtures import (
    test_config, device_manager, performance_test_data,
    measure_execution_time
)


class TestDeviceCreationAndConfiguration:
    """设备创建和配置测试（内部逻辑，可压力测试）"""
    
    def test_device_initialization_stress(self):
        """压力测试：大量设备初始化"""
        devices = []
        start_time = time.time()
        
        # 创建1000个设备
        for i in range(1000):
            room = f"房间{i % 10}"
            device_name = f"设备{i}"
            
            if i % 4 == 0:
                device = LightDevice(device_name, room)
            elif i % 4 == 1:
                device = AirConditionerDevice(device_name, room)
            elif i % 4 == 2:
                device = TVDevice(device_name, room)
            else:
                device = FanDevice(device_name, room)
            
            devices.append(device)
        
        creation_time = time.time() - start_time
        
        # 验证创建性能
        assert len(devices) == 1000
        assert creation_time < 5.0  # 1000个设备应在5秒内创建完成
        
        # 验证设备状态
        assert all(hasattr(d, 'state') for d in devices)
        assert all(d.device_type in ['灯', '空调', '电视', '风扇'] for d in devices)

    def test_device_manager_large_scale(self):
        """压力测试：大规模设备管理器"""
        manager = DeviceManager()
        
        # 清空默认设备，创建大量自定义设备
        manager.devices.clear()
        manager._index.clear()
        
        start_time = time.time()
        
        # 注册500个设备
        for i in range(500):
            room = f"房间{i // 20}"  # 每20个设备一个房间
            device_type = ['灯', '空调', '电视', '风扇'][i % 4]
            
            if device_type == '灯':
                device = LightDevice(f"灯{i}", room)
            elif device_type == '空调':
                device = AirConditionerDevice(f"空调{i}", room)
            elif device_type == '电视':
                device = TVDevice(f"电视{i}", room)
            else:
                device = FanDevice(f"风扇{i}", room)
            
            manager._register(device)
        
        registration_time = time.time() - start_time
        
        # 验证注册性能和结果
        assert len(manager.devices) == 500
        assert registration_time < 3.0  # 500个设备注册应在3秒内完成
        
        # 测试查找性能
        start_time = time.time()
        found_devices = []
        for i in range(100):
            room = f"房间{i // 4}"
            device = manager.find_device("灯", room)
            if device:
                found_devices.append(device)
        
        search_time = time.time() - start_time
        assert search_time < 1.0  # 100次查找应在1秒内完成

    def test_concurrent_device_operations(self):
        """压力测试：并发设备操作"""
        manager = DeviceManager()
        
        # 只使用有效的设备-房间组合
        valid_combinations = [
            ("灯", "客厅"), ("电视", "客厅"), ("空调", "客厅"), ("风扇", "客厅"),
            ("灯", "主卧"), ("空调", "主卧"),
            ("灯", "次卧"), ("风扇", "次卧")
        ]
        
        def operate_devices(thread_id):
            """在单个线程中操作设备"""
            results = []
            for i in range(50):  # 每个线程50次操作
                action = "打开" if i % 2 == 0 else "关闭"
                device_type, room = valid_combinations[i % len(valid_combinations)]
                
                result = manager.perform_action(action, device_type, room)
                results.append(result["success"])
            
            return sum(results)  # 返回成功次数
        
        # 10个线程并发操作
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(operate_devices, i) for i in range(10)]
            success_counts = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        total_operations = 10 * 50  # 500次操作
        total_success = sum(success_counts)
        
        # 验证并发性能
        assert total_time < 10.0  # 500次并发操作应在10秒内完成
        assert total_success >= total_operations * 0.9  # 90%成功率
        assert len(success_counts) == 10  # 所有线程都完成

    def test_device_state_consistency_stress(self):
        """压力测试：设备状态一致性"""
        manager = DeviceManager()
        
        # 大量状态变更操作
        operations = []
        for i in range(200):
            device_type = ["灯", "空调", "电视", "风扇"][i % 4]
            room = ["客厅", "主卧", "次卧"][i % 3]
            
            if device_type == "灯":
                operations.append((device_type, room, "调节", "亮度", i % 100))
            elif device_type == "空调":
                operations.append((device_type, room, "设置", "温度", 16 + (i % 15)))
            elif device_type == "电视":
                operations.append((device_type, room, "调节", "音量", i % 100))
            else:
                operations.append((device_type, room, "设置", "风速", 1 + (i % 5)))
        
        # 执行所有操作
        start_time = time.time()
        successful_ops = 0
        
        for device_type, room, action, attribute, value in operations:
            result = manager.perform_action(action, device_type, room, attribute=attribute, number_value=value)
            if result["success"]:
                successful_ops += 1
        
        execution_time = time.time() - start_time
        
        # 验证结果
        assert execution_time < 5.0  # 200次操作应在5秒内完成
        assert successful_ops >= len(operations) * 0.95  # 95%成功率
        
        # 验证状态一致性
        snapshot = manager.snapshot()
        for device_key, state in snapshot.items():
            assert isinstance(state, dict)
            assert "on" in state

    def test_device_attribute_validation_comprehensive(self):
        """全面测试设备属性验证（内部逻辑）"""
        test_cases = [
            # 灯设备测试
            (LightDevice("测试灯", "测试房间"), [
                ("亮度", 0, True, 0),
                ("亮度", 100, True, 100),
                ("亮度", -10, True, 0),  # 边界修正
                ("亮度", 150, True, 100),  # 边界修正
                ("亮度", "50", True, 50),  # 字符串转换
                ("亮度", "abc", False, None),  # 无效值
                ("亮度", None, False, None),  # None值
                ("brightness", 75, True, 75),  # 英文属性名
            ]),
            # 空调设备测试
            (AirConditionerDevice("测试空调", "测试房间"), [
                ("温度", 16, True, 16),
                ("温度", 30, True, 30),
                ("温度", 10, True, 16),  # 下限修正
                ("温度", 35, True, 30),  # 上限修正
                ("风速", 1, True, 1),
                ("风速", 5, True, 5),
                ("风速", 0, True, 1),  # 下限修正
                ("风速", 10, True, 5),  # 上限修正
                ("模式", "制冷", True, "制冷"),
                ("模式", "制热", True, "制热"),
            ]),
            # 电视设备测试
            (TVDevice("测试电视", "测试房间"), [
                ("音量", 0, True, 0),
                ("音量", 100, True, 100),
                ("音量", -5, True, 0),  # 下限修正
                ("音量", 150, True, 100),  # 上限修正
                ("频道", 1, True, 1),
                ("频道", 999, True, 999),
                ("频道", 0, True, 1),  # 下限修正
            ]),
            # 风扇设备测试
            (FanDevice("测试风扇", "测试房间"), [
                ("风速", 1, True, 1),
                ("风速", 5, True, 5),
                ("风速", 0, True, 1),  # 下限修正
                ("风速", 10, True, 5),  # 上限修正
                ("speed", 3, True, 3),  # 英文属性名
            ])
        ]
        
        total_tests = sum(len(cases) for _, cases in test_cases)
        successful_tests = 0
        
        start_time = time.time()
        
        for device, cases in test_cases:
            for attribute, value, should_succeed, expected_value in cases:
                ok, msg = device.adjust(attribute, value)
                
                if should_succeed:
                    assert ok, f"Expected success for {device.device_type}.{attribute}={value}, but got: {msg}"
                    if expected_value is not None:
                        actual_value = device.state.get(attribute) or device.state.get(attribute.lower())
                        assert actual_value == expected_value, f"Expected {expected_value}, got {actual_value}"
                    successful_tests += 1
                else:
                    assert not ok, f"Expected failure for {device.device_type}.{attribute}={value}, but succeeded"
                    successful_tests += 1
        
        test_time = time.time() - start_time
        
        # 验证测试性能
        assert test_time < 2.0  # 所有属性验证测试应在2秒内完成
        assert successful_tests == total_tests


class TestDeviceManagerQueries:
    """设备管理器查询测试（内部逻辑，可压力测试）"""
    
    def test_query_performance_stress(self, device_manager):
        """压力测试：查询性能"""
        # 执行大量查询操作
        query_operations = [
            ("query_status", {"device_type": "灯"}),
            ("query_status", {"device_type": "空调"}),
            ("query_status", {"device_type": "电视"}),
            ("query_status", {"device_type": "风扇"}),
            ("query_status", {"room": "客厅"}),
            ("query_status", {"room": "主卧"}),
            ("query_status", {"room": "次卧"}),
            ("query_status", {}),  # 查询所有设备
        ]
        
        start_time = time.time()
        successful_queries = 0
        
        # 每种查询执行100次
        for i in range(100):
            for operation, kwargs in query_operations:
                if operation == "query_status":
                    result = device_manager.query_status(**kwargs)
                    if result["success"]:
                        successful_queries += 1
        
        query_time = time.time() - start_time
        total_queries = 100 * len(query_operations)
        
        # 验证查询性能
        assert query_time < 3.0  # 800次查询应在3秒内完成
        assert successful_queries >= total_queries * 0.95  # 95%成功率

    def test_snapshot_generation_stress(self, device_manager):
        """压力测试：快照生成"""
        # 修改一些设备状态
        device_manager.perform_action("打开", "灯", "客厅")
        device_manager.perform_action("调节", "空调", "主卧", attribute="温度", number_value=25)
        device_manager.perform_action("设置", "电视", "客厅", attribute="音量", number_value=80)
        
        # 大量快照生成
        start_time = time.time()
        snapshots = []
        
        for i in range(1000):
            snapshot = device_manager.snapshot()
            snapshots.append(snapshot)
            
            # 偶尔修改状态
            if i % 100 == 0:
                device_manager.perform_action("调节", "灯", "客厅", attribute="亮度", number_value=i % 100)
        
        snapshot_time = time.time() - start_time
        
        # 验证快照性能和一致性
        assert snapshot_time < 2.0  # 1000次快照应在2秒内完成
        assert len(snapshots) == 1000
        assert all(isinstance(s, dict) for s in snapshots)
        
        # 验证快照内容
        last_snapshot = snapshots[-1]
        assert "客厅-灯" in last_snapshot
        assert "主卧-空调" in last_snapshot
        assert last_snapshot["客厅-灯"]["on"] is True

    def test_device_pattern_generation_stress(self, device_manager):
        """压力测试：设备模式生成"""
        start_time = time.time()
        
        # 大量模式生成
        patterns_list = []
        for i in range(500):
            patterns = device_manager.get_device_patterns()
            patterns_list.append(patterns)
            
            synonyms = device_manager.get_device_synonyms()
            rooms = device_manager.get_available_rooms()
            device_types = device_manager.get_available_device_types()
        
        pattern_time = time.time() - start_time
        
        # 验证性能
        assert pattern_time < 1.0  # 500次模式生成应在1秒内完成
        assert len(patterns_list) == 500
        assert all(isinstance(p, dict) for p in patterns_list)
        
        # 验证模式内容
        last_patterns = patterns_list[-1]
        assert "灯" in last_patterns
        assert "客厅灯" in last_patterns

    def test_device_lookup_algorithms(self, device_manager):
        """测试设备查找算法性能"""
        # 测试不同查找策略的性能
        lookup_strategies = [
            ("精确匹配", lambda: device_manager.find_device("灯", "客厅")),
            ("类型匹配", lambda: device_manager.find_device("灯", None)),
            ("房间匹配", lambda: [d for d in device_manager.devices if d.room == "客厅"]),
            ("全量扫描", lambda: list(device_manager.devices)),
        ]
        
        performance_results = {}
        
        for strategy_name, lookup_func in lookup_strategies:
            start_time = time.time()
            
            # 每种策略执行1000次
            for i in range(1000):
                result = lookup_func()
            
            strategy_time = time.time() - start_time
            performance_results[strategy_name] = strategy_time
        
        # 验证性能差异
        assert performance_results["精确匹配"] < 0.5  # 精确匹配应该最快
        assert performance_results["类型匹配"] < 1.0
        assert performance_results["房间匹配"] < 2.0
        assert performance_results["全量扫描"] < 3.0


class TestDeviceManagerEventSystem:
    """设备管理器事件系统测试（内部逻辑）"""
    
    def test_event_callback_stress(self, device_manager):
        """压力测试：事件回调系统"""
        received_events = []
        event_lock = threading.Lock()
        
        def event_callback(event):
            with event_lock:
                received_events.append(event)
        
        # 注册回调
        device_manager.register_callback(event_callback)
        
        # 大量设备操作触发事件
        start_time = time.time()
        operations_count = 200
        
        for i in range(operations_count):
            device_type = ["灯", "空调", "电视", "风扇"][i % 4]
            room = ["客厅", "主卧", "次卧"][i % 3]
            action = "打开" if i % 2 == 0 else "关闭"
            
            device_manager.perform_action(action, device_type, room)
        
        event_time = time.time() - start_time
        
        # 等待所有事件处理完成
        time.sleep(0.1)
        
        # 验证事件系统性能
        assert event_time < 5.0  # 200次操作应在5秒内完成
        
        with event_lock:
            # 成功的操作应该都触发了事件
            assert len(received_events) >= operations_count * 0.8  # 至少80%的操作触发事件
            
            # 验证事件结构
            for event in received_events[:10]:  # 检查前10个事件
                assert "event" in event
                assert "version" in event
                assert "timestamp" in event
                assert event["event"] == "state_changed"

    def test_version_tracking_stress(self, device_manager):
        """压力测试：版本跟踪"""
        initial_version = device_manager.get_version()
        
        # 大量状态变更
        start_time = time.time()
        for i in range(100):
            device_manager.perform_action("调节", "灯", "客厅", attribute="亮度", number_value=i % 100)
        
        version_time = time.time() - start_time
        final_version = device_manager.get_version()
        
        # 验证版本跟踪
        assert final_version > initial_version
        assert version_time < 2.0  # 100次版本更新应在2秒内完成
        
        # 验证版本递增
        version_changes = final_version - initial_version
        assert version_changes >= 100  # 每次成功操作都应增加版本号

    def test_snapshot_diff_performance(self, device_manager):
        """测试快照差异计算性能"""
        # 创建初始快照
        snapshot1 = device_manager.snapshot()
        
        # 进行一些操作
        device_manager.perform_action("打开", "灯", "客厅")
        device_manager.perform_action("调节", "空调", "主卧", attribute="温度", number_value=24)
        
        # 创建新快照
        snapshot2 = device_manager.snapshot()
        
        # 大量差异计算
        start_time = time.time()
        diffs = []
        
        for i in range(1000):
            diff = device_manager.snapshot_diff(snapshot1, snapshot2)
            diffs.append(diff)
        
        diff_time = time.time() - start_time
        
        # 验证差异计算性能
        assert diff_time < 1.0  # 1000次差异计算应在1秒内完成
        assert len(diffs) == 1000
        
        # 验证差异内容
        sample_diff = diffs[0]
        assert "changed" in sample_diff
        assert "added" in sample_diff
        assert "removed" in sample_diff


class TestDeviceManagerEdgeCases:
    """设备管理器边界情况测试（内部逻辑）"""
    
    def test_invalid_operations_comprehensive(self, device_manager):
        """全面测试无效操作处理"""
        invalid_cases = [
            # 空值测试
            ("", "", ""),
            (None, "灯", "客厅"),
            ("打开", None, "客厅"),
            ("打开", "灯", None),
            
            # 不存在的设备/房间
            ("打开", "洗衣机", "客厅"),
            ("打开", "灯", "厨房"),
            ("打开", "不存在设备", "不存在房间"),
            
            # 无效操作
            ("跳舞", "灯", "客厅"),
            ("唱歌", "空调", "主卧"),
            
            # 特殊字符
            ("@#$", "灯", "客厅"),
            ("打开", "@#$", "客厅"),
            ("打开", "灯", "@#$"),
            
            # 过长字符串
            ("打开" * 100, "灯", "客厅"),
            ("打开", "灯" * 100, "客厅"),
            ("打开", "灯", "客厅" * 100),
        ]
        
        start_time = time.time()
        handled_cases = 0
        
        for action, device_type, room in invalid_cases:
            try:
                result = device_manager.perform_action(action, device_type, room)
                # 无效操作应该返回失败结果，而不是抛出异常
                assert not result["success"]
                assert "message" in result
                handled_cases += 1
            except Exception as e:
                # 记录异常但继续测试
                print(f"Exception for {action}, {device_type}, {room}: {e}")
        
        handling_time = time.time() - start_time
        
        # 验证错误处理性能和覆盖率
        assert handling_time < 2.0  # 错误处理应该很快
        assert handled_cases >= len(invalid_cases) * 0.8  # 至少80%的情况被正确处理

    def test_boundary_value_comprehensive(self):
        """全面边界值测试"""
        devices = [
            LightDevice("测试灯", "测试房间"),
            AirConditionerDevice("测试空调", "测试房间"),
            TVDevice("测试电视", "测试房间"),
            FanDevice("测试风扇", "测试房间")
        ]
        
        boundary_tests = [
            # 数值边界
            ("亮度", [-1000, -1, 0, 1, 50, 99, 100, 101, 1000]),
            ("温度", [-100, 15, 16, 25, 30, 31, 100]),
            ("风速", [-10, 0, 1, 3, 5, 6, 20]),
            ("音量", [-50, -1, 0, 50, 100, 101, 200]),
            ("频道", [-10, 0, 1, 50, 999, 1000]),
            
            # 字符串边界
            ("模式", ["", "制冷", "制热", "自动", "invalid_mode", "a" * 100]),
            
            # 特殊类型
            ("亮度", [None, "", "abc", "50", "50.5", True, False, [], {}]),
        ]
        
        start_time = time.time()
        total_tests = 0
        successful_tests = 0
        
        for device in devices:
            for attribute, test_values in boundary_tests:
                # 只测试设备支持的属性
                if attribute == "亮度" and not isinstance(device, LightDevice):
                    continue
                if attribute == "温度" and not isinstance(device, AirConditionerDevice):
                    continue
                if attribute in ["音量", "频道"] and not isinstance(device, TVDevice):
                    continue
                if attribute == "风速" and not isinstance(device, (AirConditionerDevice, FanDevice)):
                    continue
                if attribute == "模式" and not isinstance(device, AirConditionerDevice):
                    continue
                
                for value in test_values:
                    total_tests += 1
                    try:
                        ok, msg = device.adjust(attribute, value)
                        # 不管成功失败，只要不抛异常就算处理成功
                        successful_tests += 1
                    except Exception as e:
                        print(f"Exception in boundary test {device.device_type}.{attribute}={value}: {e}")
        
        test_time = time.time() - start_time
        
        # 验证边界测试性能
        assert test_time < 3.0  # 边界测试应在3秒内完成
        assert successful_tests >= total_tests * 0.9  # 90%的边界情况被正确处理

    def test_concurrent_state_access(self, device_manager):
        """测试并发状态访问"""
        def read_states(iterations):
            """读取设备状态"""
            results = []
            for i in range(iterations):
                snapshot = device_manager.snapshot()
                results.append(len(snapshot))
            return results
        
        def modify_states(iterations):
            """修改设备状态"""
            results = []
            for i in range(iterations):
                result = device_manager.perform_action(
                    "调节", "灯", "客厅", 
                    attribute="亮度", 
                    number_value=i % 100
                )
                results.append(result["success"])
            return results
        
        # 并发读写测试
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            # 3个读线程，3个写线程
            read_futures = [executor.submit(read_states, 50) for _ in range(3)]
            write_futures = [executor.submit(modify_states, 50) for _ in range(3)]
            
            # 等待所有任务完成
            read_results = [future.result() for future in read_futures]
            write_results = [future.result() for future in write_futures]
        
        concurrent_time = time.time() - start_time
        
        # 验证并发访问
        assert concurrent_time < 5.0  # 并发操作应在5秒内完成
        assert len(read_results) == 3
        assert len(write_results) == 3
        
        # 验证数据一致性（没有崩溃或异常结果）
        for read_result in read_results:
            assert all(count > 0 for count in read_result)  # 始终有设备存在
        
        for write_result in write_results:
            assert sum(write_result) >= len(write_result) * 0.8  # 80%写操作成功


@pytest.mark.slow
class TestDeviceManagerMemoryAndResources:
    """设备管理器内存和资源测试（内部逻辑，可压力测试）"""
    
    def test_memory_leak_detection(self):
        """检测内存泄漏"""
        import gc
        import sys
        
        # 强制垃圾回收
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # 创建和销毁大量设备管理器
        for i in range(100):
            manager = DeviceManager()
            
            # 进行一些操作
            for j in range(10):
                manager.perform_action("打开", "灯", "客厅")
                manager.perform_action("关闭", "灯", "客厅")
            
            # 模拟销毁
            manager.devices.clear()
            manager._index.clear()
            manager._callbacks.clear()
            del manager
        
        # 强制垃圾回收
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # 检查对象数量增长
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"可能存在内存泄漏，对象增长: {object_growth}"

    def test_callback_memory_management(self):
        """测试回调内存管理"""
        manager = DeviceManager()
        
        # 注册大量回调
        callbacks = []
        for i in range(1000):
            def callback(event, index=i):
                pass  # 空回调
            
            callbacks.append(callback)
            manager.register_callback(callback)
        
        # 验证回调注册
        assert len(manager._callbacks) == 1000
        
        # 触发一些事件
        for i in range(10):
            manager.perform_action("打开", "灯", "客厅")
        
        # 注销所有回调
        for callback in callbacks:
            manager.unregister_callback(callback)
        
        # 验证回调清理
        assert len(manager._callbacks) == 0
        
        # 再次触发事件，应该没有回调被执行
        manager.perform_action("关闭", "灯", "客厅")

    def test_large_state_management(self):
        """测试大状态管理"""
        manager = DeviceManager()
        
        # 清空默认设备
        manager.devices.clear()
        manager._index.clear()
        
        # 创建大量设备和复杂状态
        for i in range(200):
            room = f"房间{i // 10}"
            
            # 创建复杂状态的设备
            light = LightDevice(f"智能灯{i}", room)
            light.state.update({
                "color": f"rgb({i%256},{(i*2)%256},{(i*3)%256})",
                "schedule": [f"定时{j}" for j in range(10)],
                "history": [f"操作{k}" for k in range(20)],
                "metadata": {"id": i, "model": f"model_{i}", "firmware": f"v{i%10}.{i%100}"}
            })
            
            manager._register(light)
        
        # 测试大状态快照性能
        start_time = time.time()
        snapshot = manager.snapshot()
        snapshot_time = time.time() - start_time
        
        assert snapshot_time < 2.0  # 大状态快照应在2秒内完成
        assert len(snapshot) == 200
        
        # 测试大状态操作性能
        start_time = time.time()
        for i in range(50):
            result = manager.perform_action("调节", f"智能灯{i}", f"房间{i//10}", attribute="亮度", number_value=i % 100)
            assert result["success"]
        
        operation_time = time.time() - start_time
        assert operation_time < 3.0  # 50次复杂设备操作应在3秒内完成


# 配置pytest标记
pytestmark = [
    pytest.mark.device_manager,
    pytest.mark.internal_logic  # 标记为内部逻辑测试，可以压力测试
]
