#!/usr/bin/env python3
"""离线测试代词消解功能"""

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

def test_pronoun_resolution_offline():
    """离线测试代词消解功能"""
    config = SimpleConfig()
    recognizer = IntentRecognizer(config)
    
    # 测试输入
    test_input = "把它的温度调到26度"
    
    print("=== 离线代词消解测试 ===")
    print(f"输入: {test_input}")
    
    # 直接调用_preprocess_input方法
    processed_input, pronoun_info = recognizer._preprocess_input(test_input)
    
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
                
                # 验证替换是否正确
                expected_output = "把空调的温度调到26度"
                if final_input == expected_output:
                    print("✅ 测试通过：代词消解功能正常")
                    return True
                else:
                    print(f"❌ 测试失败：期望 '{expected_output}'，实际 '{final_input}'")
                    return False
    
    print("❌ 测试失败：未检测到代词或焦点实体")
    return False

if __name__ == "__main__":
    success = test_pronoun_resolution_offline()
    sys.exit(0 if success else 1)