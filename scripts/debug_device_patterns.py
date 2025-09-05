#!/usr/bin/env python3
"""测试设备模式生成"""

from dialogue_manager.device_manager import DeviceManager

def test_device_patterns():
    device_manager = DeviceManager()
    patterns = device_manager.get_device_patterns()
    print("设备模式:")
    for pattern, device_type in patterns.items():
        print(f"  {pattern} -> {device_type}")

if __name__ == "__main__":
    test_device_patterns()
