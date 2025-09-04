#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dialogue_manager.engine import DialogueEngine, EngineConfig
from dialogue_manager.intent import IntentRecognizer

def debug_you_are_who():
    """调试'你是谁'的意图识别问题"""
    # 创建配置
    config = EngineConfig(
        enable_context_entity_fill=True,
        focus_switch_policy="conservative",
        enable_device_manager=True,
        focus_entity_turn_decay=2
    )
    
    # 创建对话引擎
    engine = DialogueEngine("fake_api_key", config)
    
    # 开始会话
    engine.start_session('test_user')
    
    # 第一轮：建立焦点
    print("=== 第一轮：建立焦点 ===")
    response, debug_info = engine.process_input('打开客厅的灯')
    print(f"焦点实体: {engine.context.get('current_focus', '无')}")
    
    # 第二轮：不涉及焦点实体的对话
    print("\n=== 第二轮：不涉及焦点实体 ===")
    response, debug_info = engine.process_input('今天天气怎么样')
    print(f"焦点实体: {engine.context.get('current_focus', '无')}")
    
    # 第三轮：详细调试"你是谁"
    print("\n=== 第三轮：详细调试'你是谁' ===")
    
    # 手动调用意图识别器
    recognizer = IntentRecognizer(config)
    user_input = "你是谁"
    context = engine.context
    history = engine.dialogue_history
    
    print(f"用户输入: '{user_input}'")
    print(f"当前上下文中的焦点: {context.get('current_focus', '无')}")
    
    # 步骤1：基础意图分类
    intent_scores = recognizer._classify_intent(user_input)
    print(f"基础意图得分: {intent_scores}")
    
    # 步骤2：上下文增强
    enhanced_scores = recognizer._enhance_with_context(user_input, intent_scores.copy(), context, history)
    print(f"上下文增强后得分: {enhanced_scores}")
    
    # 步骤3：选择最佳意图
    best_intent, confidence = recognizer._select_best_intent(enhanced_scores)
    print(f"最佳意图: {best_intent}, 置信度: {confidence}")
    
    # 步骤4：实体抽取
    entities = recognizer._extract_entities_with_context(user_input, best_intent, context)
    print(f"提取的实体: {[(e.value, e.entity_type) for e in entities]}")
    
    # 现在通过引擎处理
    print("\n--- 通过引擎处理 ---")
    response, debug_info = engine.process_input('你是谁')
    print(f"引擎识别的意图: {debug_info['intent_result']['intent']}")
    print(f"引擎识别的置信度: {debug_info['intent_result']['confidence']}")
    print(f"引擎提取的实体: {[(e['value'], e['entity_type']) for e in debug_info['intent_result']['entities']]}")
    print(f"处理后的焦点实体: {engine.context.get('current_focus', '无')}")

if __name__ == "__main__":
    debug_you_are_who()