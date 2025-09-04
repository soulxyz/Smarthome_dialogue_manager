#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dialogue_manager.engine import DialogueEngine, EngineConfig
from dialogue_manager.device_manager import DeviceManager

def debug_focus_issue():
    """调试焦点实体问题"""
    # 创建配置
    config = EngineConfig(
        enable_context_entity_fill=True,
        focus_switch_policy="conservative",
        enable_device_manager=True
    )
    
    # 创建对话引擎（需要api_key作为第一个参数）
    engine = DialogueEngine("fake_api_key", config)
    
    # 开始会话
    engine.start_session('test_user')
    
    # 测试只有设备名的输入
    print("=== 测试输入: '空调' ===")
    response, debug_info = engine.process_input('空调')
    
    print(f"响应: {response}")
    print(f"意图: {debug_info['intent_result']['intent']}")
    print(f"置信度: {debug_info['intent_result']['confidence']}")
    print(f"实体: {debug_info['intent_result']['entities']}")
    print(f"上下文: {engine.context}")
    
    # 检查焦点实体
    if 'current_focus' in engine.context:
        print(f"焦点实体: {engine.context['current_focus']}")
    else:
        print("没有焦点实体")

if __name__ == "__main__":
    debug_focus_issue()