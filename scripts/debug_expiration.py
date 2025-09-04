#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dialogue_manager.engine import DialogueEngine, EngineConfig

def debug_expiration():
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
    if 'current_focus' in engine.context:
        focus = engine.context['current_focus']
        print(f"焦点实体: {focus['value']}, turn_count: {focus['turn_count']}")
    else:
        print("没有焦点实体")
    
    # 第三轮：再次不涉及焦点实体的对话
    print("\n=== 第三轮：再次不涉及焦点实体 ===")
    response, debug_info = engine.process_input('你是谁')
    print(f"响应: {response}")
    print(f"识别的意图: {debug_info['intent_result']['intent']}")
    print(f"意图置信度: {debug_info['intent_result']['confidence']}")
    entities = debug_info['intent_result']['entities']
    print(f"提取的实体: {[(e['value'], e['entity_type']) for e in entities]}")
    if 'current_focus' in engine.context:
        focus = engine.context['current_focus']
        print(f"焦点实体: {focus['value']}, turn_count: {focus['turn_count']}")
    else:
        print("焦点实体已过期")
    
    # 第四轮：测试省略主语的指令
    print("\n=== 第四轮：测试省略主语 ===")
    response, debug_info = engine.process_input('关掉')
    print(f"响应: {response}")
    entities = debug_info['intent_result']['entities']
    device_entities = [e for e in entities if e['entity_type'] == 'device']
    print(f"设备实体数量: {len(device_entities)}")
    if device_entities:
        print(f"设备实体: {[e['value'] for e in device_entities]}")
    else:
        print("没有设备实体（符合预期，因为焦点已过期）")

if __name__ == "__main__":
    debug_expiration()