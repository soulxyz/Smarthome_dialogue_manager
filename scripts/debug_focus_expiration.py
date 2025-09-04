#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dialogue_manager.engine import DialogueEngine, EngineConfig
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

def debug_focus_expiration():
    """调试焦点实体过期问题"""
    # 创建配置
    config = EngineConfig(
        enable_context_entity_fill=True,
        focus_switch_policy="conservative",
        enable_device_manager=True,
        focus_entity_turn_decay=2  # 设置为2轮后过期
    )
    
    # 创建对话引擎
    engine = DialogueEngine("fake_api_key", config)
    
    # 开始会话
    engine.start_session('test_user')
    
    # 第一轮：建立焦点
    print("=== 第一轮：建立焦点 ===")
    response, debug_info = engine.process_input('打开客厅的灯')
    print(f"响应: {response}")
    if 'current_focus' in engine.context:
        focus = engine.context['current_focus']
        print(f"焦点实体: {focus['value']}, turn_count: {focus['turn_count']}")
    else:
        print("没有焦点实体")
    
    # 第二轮：不涉及焦点实体的对话
    print("\n=== 第二轮：不涉及焦点实体 ===")
    response, debug_info = engine.process_input('今天天气怎么样')
    print(f"响应: {response}")
    print(f"意图: {debug_info['intent_result']['intent']}")
    print(f"实体: {[(e['value'], e['entity_type']) for e in debug_info['intent_result']['entities']]}")
    if 'current_focus' in engine.context:
        focus = engine.context['current_focus']
        print(f"焦点实体: {focus['value']}, turn_count: {focus['turn_count']}")
    else:
        print("没有焦点实体")
    
    # 第三轮：再次不涉及焦点实体的对话
    print("\n=== 第三轮：再次不涉及焦点实体 ===")
    response, debug_info = engine.process_input('你是谁')
    print(f"响应: {response}")
    print(f"意图: {debug_info['intent_result']['intent']}")
    print(f"实体: {[(e['value'], e['entity_type']) for e in debug_info['intent_result']['entities']]}")
    if 'current_focus' in engine.context:
        focus = engine.context['current_focus']
        print(f"焦点实体: {focus['value']}, turn_count: {focus['turn_count']}")
    else:
        print("焦点实体已过期")
    
    # 打印完整上下文
    print("\n=== 完整上下文 ===")
    print(engine.context)

if __name__ == "__main__":
    debug_focus_expiration()