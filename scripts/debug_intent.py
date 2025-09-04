#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dialogue_manager.intent import IntentRecognizer
from dialogue_manager.engine import EngineConfig

def debug_intent():
    """调试意图识别问题"""
    # 创建配置
    config = EngineConfig(
        enable_context_entity_fill=True,
        focus_switch_policy="conservative",
        enable_device_manager=True,
        focus_entity_turn_decay=2
    )
    
    # 创建意图识别器
    recognizer = IntentRecognizer(config)
    
    # 测试"你是谁"的意图识别
    test_inputs = ["你是谁", "今天天气怎么样", "关掉", "空调"]
    
    for user_input in test_inputs:
        print(f"\n=== 测试输入: '{user_input}' ===")
        
        # 获取意图得分
        scores = recognizer._classify_intent(user_input)
        print(f"意图得分: {scores}")
        
        # 获取最佳意图
        best_intent = max(scores.items(), key=lambda x: x[1])
        print(f"最佳意图: {best_intent[0]} (得分: {best_intent[1]})")
        
        # 提取实体
        entities = recognizer._extract_entities(user_input)
        print(f"提取的实体: {[(e.value, e.entity_type) for e in entities]}")

if __name__ == "__main__":
    debug_intent()