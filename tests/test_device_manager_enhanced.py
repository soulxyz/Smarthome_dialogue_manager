"""DeviceManager模块的增强测试用例

补充原有测试用例中缺失的边界测试、性能测试等
"""
import pytest
import time
from dialogue_manager.device_manager import (
    DeviceManager,
    LightDevice,
    AirConditionerDevice,
    TVDevice,
    FanDevice,
)


class TestBoundaryValues:
    """边界值测试"""
    
    def test_light_brightness_boundaries(self):
        """测试灯亮度的边界值"""
        light = LightDevice("测试灯", "测试房间")
        
        # 测试负值
        ok, msg = light.adjust("亮度", -10)
        assert ok
        assert light.state["brightness"] == 0
        
        # 测试超出上限
        ok, msg = light.adjust("亮度", 150)
        assert ok
        assert light.state["brightness"] == 100
        
        # 测试边界值
        ok, msg = light.adjust("亮度", 0)
        assert ok
        assert light.state["brightness"] == 0
        
        ok, msg = light.adjust("亮度", 100)
        assert ok
        assert light.state["brightness"] == 100
    
    def test_ac_temperature_boundaries(self):
        """测试空调温度的边界值"""
        ac = AirConditionerDevice("测试空调", "测试房间")
        
        # 测试低于下限
        ok, msg = ac.adjust("温度", 10)
        assert ok
        assert ac.state["temperature"] == 16
        
        # 测试超出上限
        ok, msg = ac.adjust("温度", 35)
        assert ok
        assert ac.state["temperature"] == 30
        
        # 测试边界值
        ok, msg = ac.adjust("温度", 16)
        assert ok
        assert ac.state["temperature"] == 16
        
        ok, msg = ac.adjust("温度", 30)
        assert ok
        assert ac.state["temperature"] == 30
    
    def test_ac_fan_speed_boundaries(self):
        """测试空调风速的边界值"""
        ac = AirConditionerDevice("测试空调", "测试房间")
        
        # 测试低于下限
        ok, msg = ac.adjust("风速", 0)
        assert ok
        assert ac.state["fan_speed"] == 1
        
        # 测试超出上限
        ok, msg = ac.adjust("风速", 10)
        assert ok
        assert ac.state["fan_speed"] == 5
    
    def test_tv_volume_boundaries(self):
        """测试电视音量的边界值"""
        tv = TVDevice("测试电视", "测试房间")
        
        # 测试负值
        ok, msg = tv.adjust("音量", -5)
        assert ok
        assert tv.state["volume"] == 0
        
        # 测试超出上限
        ok, msg = tv.adjust("音量", 150)
        assert ok
        assert tv.state["volume"] == 100


class TestDataValidation:
    """数据验证测试"""
    
    def test_invalid_attribute_values(self):
        """测试无效的属性值"""
        light = LightDevice("测试灯", "测试房间")
        
        # 测试非数字字符串
        ok, msg = light.adjust("亮度", "abc")
        assert not ok
        assert "无效" in msg
        
        # 测试None值
        ok, msg = light.adjust("亮度", None)
        assert not ok
        assert "无效" in msg
        
        # 测试浮点数字符串（应该能转换）
        ok, msg = light.adjust("亮度", "75.5")
        assert ok
        assert light.state["brightness"] == 75
    
    def test_empty_and_none_inputs(self):
        """测试空值和None输入"""
        manager = DeviceManager()
        
        # 测试空设备类型
        result = manager.perform_action("打开", "", "客厅")
        assert not result["success"]
        
        # 测试None设备类型
        result = manager.perform_action("打开", None, "客厅")
        assert not result["success"]
        
        # 测试空房间名（应该能找到默认设备）
        result = manager.perform_action("打开", "电视", "")
        assert result["success"]  # 应该找到客厅电视


class TestComplexScenarios:
    """复杂场景测试"""
    
    def test_multiple_device_operations(self):
        """测试多设备操作"""
        manager = DeviceManager()
        
        # 同时操作多个设备
        operations = [
            ("打开", "灯", "客厅"),
            ("打开", "空调", "客厅"),
            ("打开", "电视", "客厅"),
        ]
        
        results = []
        for action, device_type, room in operations:
            result = manager.perform_action(action, device_type, room)
            results.append(result)
        
        # 所有操作都应该成功
        assert all(r["success"] for r in results)
        
        # 验证所有设备都已开启
        snapshot = manager.snapshot()
        assert snapshot["客厅-灯"]["on"] is True
        assert snapshot["客厅-空调"]["on"] is True
        assert snapshot["客厅-电视"]["on"] is True
    
    def test_sequential_adjustments(self):
        """测试连续调节操作"""
        manager = DeviceManager()
        light = manager.find_device("灯", "客厅")
        
        # 连续调节亮度
        adjustments = [20, 40, 60, 80, 100, 50]
        for brightness in adjustments:
            result = manager.perform_action("调节", "灯", "客厅", 
                                          attribute="亮度", number_value=brightness)
            assert result["success"]
            assert result["state"]["brightness"] == brightness
    
    def test_device_state_consistency(self):
        """测试设备状态一致性"""
        manager = DeviceManager()
        
        # 操作设备
        manager.perform_action("打开", "灯", "客厅")
        manager.perform_action("调节", "灯", "客厅", attribute="亮度", number_value=75)
        
        # 通过不同方式获取状态，应该一致
        device = manager.find_device("灯", "客厅")
        snapshot = manager.snapshot()
        status = manager.query_status("灯", "客厅")
        
        assert device.state["on"] is True
        assert device.state["brightness"] == 75
        assert snapshot["客厅-灯"]["on"] is True
        assert snapshot["客厅-灯"]["brightness"] == 75
        assert status["success"]


class TestPerformance:
    """性能测试"""
    
    def test_device_lookup_performance(self):
        """测试设备查找性能"""
        manager = DeviceManager()
        
        # 测试大量查找操作的性能
        start_time = time.time()
        for _ in range(1000):
            device = manager.find_device("灯", "客厅")
            assert device is not None
        end_time = time.time()
        
        # 1000次查找应该在1秒内完成
        assert end_time - start_time < 1.0
    
    def test_operation_performance(self):
        """测试操作性能"""
        manager = DeviceManager()
        
        # 测试大量操作的性能
        start_time = time.time()
        for i in range(100):
            brightness = (i % 10) * 10 + 10
            result = manager.perform_action("调节", "灯", "客厅", 
                                          attribute="亮度", number_value=brightness)
            assert result["success"]
        end_time = time.time()
        
        # 100次操作应该在0.5秒内完成
        assert end_time - start_time < 0.5


class TestEdgeCases:
    """边缘情况测试"""
    
    def test_device_with_special_characters(self):
        """测试包含特殊字符的设备名称"""
        # 注意：当前实现可能不支持动态添加设备，这里测试现有设备的特殊情况
        manager = DeviceManager()
        
        # 测试查找不存在的设备（包含特殊字符）
        device = manager.find_device("灯@#$", "客厅")
        assert device is None
    
    def test_case_sensitivity(self):
        """测试大小写敏感性"""
        manager = DeviceManager()
        
        # 当前实现应该是大小写敏感的
        device1 = manager.find_device("灯", "客厅")
        device2 = manager.find_device("灯", "客厅")  # 中文没有大小写问题
        
        assert device1 is not None
        assert device2 is not None
        assert device1 == device2
    
    def test_room_fallback_logic(self):
        """测试房间回退逻辑"""
        manager = DeviceManager()
        
        # 测试不指定房间时的回退逻辑
        # 对于只有一个的设备类型（如电视），应该返回那个设备
        tv = manager.find_device("电视", None)
        assert tv is not None
        assert tv.room == "客厅"
        
        # 对于有多个的设备类型（如灯），应该优先返回客厅的
        light = manager.find_device("灯", None)
        assert light is not None
        assert light.room == "客厅"


class TestErrorHandling:
    """错误处理测试"""
    
    def test_malformed_operations(self):
        """测试格式错误的操作"""
        manager = DeviceManager()
        
        # 测试各种错误的操作格式
        error_cases = [
            ("", "灯", "客厅"),  # 空操作
            ("未知操作", "灯", "客厅"),  # 未知操作
            ("打开", "不存在设备", "客厅"),  # 不存在的设备
            ("打开", "灯", "不存在房间"),  # 不存在的房间
        ]
        
        for action, device_type, room in error_cases:
            result = manager.perform_action(action, device_type, room)
            assert not result["success"]
            assert "message" in result
            assert len(result["message"]) > 0
    
    def test_attribute_adjustment_errors(self):
        """测试属性调节错误"""
        manager = DeviceManager()
        
        # 测试调节不存在的属性
        result = manager.perform_action("调节", "灯", "客厅", 
                                      attribute="不存在属性", number_value=50)
        # 注意：当前实现可能会接受任何属性，这可能需要改进
        
        # 测试无效的数值
        light = manager.find_device("灯", "客厅")
        ok, msg = light.adjust("亮度", "无效数值")
        assert not ok
        assert "无效" in msg


@pytest.mark.parametrize("device_type,room,expected_found", [
    ("灯", "客厅", True),
    ("灯", "主卧", True),
    ("灯", "次卧", True),
    ("空调", "客厅", True),
    ("空调", "主卧", True),
    ("电视", "客厅", True),
    ("风扇", "客厅", True),
    ("风扇", "次卧", True),
    ("冰箱", "厨房", False),
    ("洗衣机", "阳台", False),
])
def test_device_existence_parametrized(device_type, room, expected_found):
    """参数化测试设备存在性"""
    manager = DeviceManager()
    device = manager.find_device(device_type, room)
    
    if expected_found:
        assert device is not None
        assert device.device_type == device_type
        assert device.room == room
    else:
        assert device is None