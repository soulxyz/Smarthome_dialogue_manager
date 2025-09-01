import pytest
from dialogue_manager.device_manager import (
    DeviceManager,
    LightDevice,
    AirConditionerDevice,
    TVDevice,
    FanDevice,
)

@pytest.fixture
def manager():
    """Provides a DeviceManager instance for tests."""
    return DeviceManager()

def test_initial_devices(manager: DeviceManager):
    """Tests that the default devices are created."""
    assert len(manager.devices) == 8
    assert manager.find_device("灯", "客厅") is not None
    assert manager.find_device("电视", "客厅") is not None
    assert manager.find_device("空调", "客厅") is not None
    assert manager.find_device("风扇", "客厅") is not None
    assert manager.find_device("灯", "主卧") is not None
    assert manager.find_device("空调", "主卧") is not None
    assert manager.find_device("灯", "次卧") is not None
    assert manager.find_device("风扇", "次卧") is not None

def test_find_device(manager: DeviceManager):
    """Tests the find_device method."""
    # Find by type and room
    light = manager.find_device("灯", "客厅")
    assert light is not None
    assert light.device_type == "灯"
    assert light.room == "客厅"

    # Find non-existent device
    assert manager.find_device("冰箱", "厨房") is None

    # Find by type only (when unique)
    tv = manager.find_device("电视", None)
    assert tv is not None
    assert tv.device_type == "电视"

    # Find by type only (when multiple, should prefer living room)
    ac = manager.find_device("空调", None)
    assert ac is not None
    assert ac.room == "客厅"

def test_light_device():
    light = LightDevice("测试灯", "书房")
    assert light.state["on"] is False
    assert light.state["brightness"] == 50

    ok, msg = light.turn_on()
    assert ok and "已开启" in msg
    assert light.state["on"] is True

    ok, msg = light.adjust("亮度", 80)
    assert ok and "亮度设置为80" in msg
    assert light.state["brightness"] == 80

    ok, msg = light.adjust("brightness", "invalid")
    assert not ok and "无效" in msg

    assert "亮度80%" in light.status_text()

def test_ac_device():
    ac = AirConditionerDevice("测试空调", "卧室")
    assert ac.state["temperature"] == 26

    ok, msg = ac.adjust("温度", 28)
    assert ok and "温度设置为28" in msg
    assert ac.state["temperature"] == 28

    ok, msg = ac.adjust("fan_speed", 3)
    assert ok and "风速设置为3" in msg
    assert ac.state["fan_speed"] == 3

    ok, msg = ac.adjust("mode", "制冷")
    assert ok and "模式设置为制冷" in msg
    assert ac.state["mode"] == "制冷"

def test_tv_device():
    tv = TVDevice("测试电视", "游戏室")
    assert tv.state["volume"] == 20

    ok, msg = tv.adjust("音量", 30)
    assert ok and "音量设置为30" in msg
    assert tv.state["volume"] == 30

    ok, msg = tv.adjust("channel", 10)
    assert ok and "切换到第10频道" in msg
    assert tv.state["channel"] == 10

def test_fan_device():
    fan = FanDevice("测试风扇", "阳台")
    assert fan.state["speed"] == 2

    ok, msg = fan.adjust("speed", 4)
    assert ok and "风速设置为4" in msg
    assert fan.state["speed"] == 4

def test_perform_action_turn_on_off(manager: DeviceManager):
    """Tests turning devices on and off via perform_action."""
    # Turn on living room light
    result = manager.perform_action("打开", "灯", "客厅")
    assert result["success"]
    assert "已开启" in result["message"]
    assert result["device_type"] == "灯"
    assert result["room"] == "客厅"
    assert result["state"]["on"] is True

    # Turn off the same light
    result = manager.perform_action("关闭", "灯", "客厅")
    assert result["success"]
    assert "已关闭" in result["message"]
    assert result["state"]["on"] is False

def test_perform_action_adjust(manager: DeviceManager):
    """Tests adjusting device attributes via perform_action."""
    # Adjust brightness
    result = manager.perform_action("调节", "灯", "客厅", attribute="亮度", number_value=75)
    assert result["success"]
    assert "亮度设置为75" in result["message"]
    assert result["state"]["brightness"] == 75

    # Adjust temperature
    result = manager.perform_action("设置", "空调", "主卧", attribute="温度", number_value=22)
    assert result["success"]
    assert "温度设置为22" in result["message"]
    assert result["state"]["temperature"] == 22

def test_perform_action_increase_decrease(manager: DeviceManager):
    """Tests increasing and decreasing device attributes."""
    # Increase AC temperature
    ac = manager.find_device("空调", "客厅")
    initial_temp = ac.state["temperature"]
    result = manager.perform_action("调高", "空调", "客厅", attribute="温度")
    assert result["success"]
    assert result["state"]["temperature"] == initial_temp + 5

    # Decrease light brightness
    light = manager.find_device("灯", "次卧")
    light.state["brightness"] = 50
    result = manager.perform_action("调低", "灯", "次卧", attribute="亮度")
    assert result["success"]
    assert result["state"]["brightness"] == 45

def test_query_status(manager: DeviceManager):
    """Tests the query_status method."""
    # Query a specific device
    result = manager.query_status("电视", "客厅")
    assert result["success"]
    assert "客厅电视" in result["message"]
    assert len(result["states"]) == 1

    # Query all devices of a type
    result = manager.query_status("灯")
    assert result["success"]
    assert "客厅灯" in result["message"]
    assert "主卧灯" in result["message"]
    assert "次卧灯" in result["message"]
    assert len(result["states"]) == 3

    # Query all devices
    result = manager.query_status()
    assert result["success"]
    assert len(result["states"]) == 8

def test_error_scenarios(manager: DeviceManager):
    """Tests various error and edge cases."""
    # Action on non-existent device
    result = manager.perform_action("打开", "冰箱", "厨房")
    assert not result["success"]
    assert "未找到" in result["message"]

    # Unsupported action
    result = manager.perform_action("跳舞", "灯", "客厅")
    assert not result["success"]
    assert "不支持" in result["message"]

    # Invalid adjustment value
    light = manager.find_device("灯", "客厅")
    ok, msg = light.adjust("亮度", "abc")
    assert not ok
    assert "无效" in msg

def test_snapshot(manager: DeviceManager):
    """Tests the snapshot method."""
    snapshot = manager.snapshot()
    assert "客厅-灯" in snapshot
    assert "主卧-空调" in snapshot
    assert snapshot["客厅-电视"]["on"] is False