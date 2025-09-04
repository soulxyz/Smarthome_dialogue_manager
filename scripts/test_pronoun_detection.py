#!/usr/bin/env python3
"""简单的代词检测测试脚本"""

import sys
import os

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# 创建一个简单的配置对象
class SimpleConfig:
    def __init__(self):
        self.device_patterns = {}

# 创建一个简单的IntentRecognizer来测试代词检测
from dialogue_manager.intent import IntentRecognizer

def test_pronoun_detection():
    """测试代词检测功能"""
    config = SimpleConfig()
    recognizer = IntentRecognizer(config)
    
    # 测试输入
    test_inputs = [
        "把它的温度调到26度",
        "打开它",
        "关闭那个设备",
        "调节它的亮度",
        "打开空调"  # 没有代词的对照
    ]
    
    print("=== 代词检测测试 ===")
    for user_input in test_inputs:
        print(f"\n输入: {user_input}")
        
        # 直接调用_preprocess_input方法
        processed_input, pronoun_info = recognizer._preprocess_input(user_input)
        
        print(f"处理后: {processed_input}")
        print(f"代词信息: {pronoun_info}")
        
        # 测试代词替换
        if pronoun_info["has_pronoun"]:
            context = {"current_focus": {"value": "空调"}}
            if pronoun_info["has_pronoun"] and "current_focus" in context:
                focus_entity = context["current_focus"].get("value")
                if focus_entity:
                    final_input = processed_input.replace(pronoun_info["placeholder"], focus_entity)
                    print(f"替换后: {final_input}")

if __name__ == "__main__":
    test_pronoun_detection()