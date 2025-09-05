#!/usr/bin/env python3
"""调试失败案例脚本

专门用于分析语义覆盖度测试中的失败案例
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dialogue_manager.engine import DialogueEngine, EngineConfig
from tests.config import get_test_config

def debug_specific_cases():
    """调试特定的失败案例"""
    
    # 获取配置
    try:
        config = get_test_config()
        if config.api_key and config.api_key != "sk-test-key-for-offline-testing":
            api_key = config.api_key
            print("✅ 使用测试配置中的API密钥")
        else:
            print("❌ 没有找到有效的API密钥")
            return
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return
    
    # 创建引擎
    engine_config = EngineConfig(
        confidence_threshold=0.7,
        enable_clarification=True,
        enable_device_manager=True,
        max_turns=10
    )
    
    engine = DialogueEngine(api_key, engine_config)
    
    # 失败案例分析
    failure_cases = [
        {
            "input": "温度多少?",
            "expected": "query_status",
            "description": "温度查询被误识别为设备控制"
        },
        {
            "input": "空调开着没?", 
            "expected": "query_status",
            "description": "设备状态查询被误识别为设备控制"
        },
        {
            "input": "看个电影",
            "expected": "scene_control", 
            "description": "场景控制被误识别为设备控制"
        },
        {
            "input": "声音大一点",
            "expected": "device_control",
            "description": "音量调节完全无法识别"
        }
    ]
    
    print("🔍 失败案例详细分析")
    print("=" * 60)
    
    for i, case in enumerate(failure_cases, 1):
        print(f"\n📋 案例 {i}: {case['description']}")
        print(f"输入: '{case['input']}'")
        print(f"期望意图: {case['expected']}")
        print("-" * 40)
        
        # 重置引擎状态
        engine.dialogue_history.clear()
        engine.context.clear()
        
        # 处理输入
        response, debug_info = engine.process_input(case['input'])
        
        # 分析结果
        intent_result = debug_info.get("intent_result", {})
        recognized_intent = intent_result.get("intent", "unknown")
        confidence = intent_result.get("confidence", 0.0)
        need_clarification = intent_result.get("need_clarification", False)
        entities = intent_result.get("entities", [])
        
        print(f"实际意图: {recognized_intent}")
        print(f"置信度: {confidence:.3f}")
        print(f"需要澄清: {need_clarification}")
        print(f"识别实体: {[e.get('name', e.get('value', '')) for e in entities]}")
        
        # 分析澄清过程
        clarification_result = debug_info.get("clarification_result")
        if clarification_result:
            print(f"澄清结果: {clarification_result.get('intent')} (conf:{clarification_result.get('confidence', 0):.3f})")
            print(f"澄清文本: {clarification_result.get('clarified_text', 'N/A')}")
        
        clarification_candidates = debug_info.get("clarification_candidates", [])
        if clarification_candidates:
            print(f"澄清候选: {clarification_candidates}")
        
        # 判断结果
        is_correct = recognized_intent == case['expected']
        result_icon = "✅" if is_correct else "❌"
        print(f"结果: {result_icon} {'正确' if is_correct else '失败'}")
        
        if not is_correct:
            print("🔍 失败原因分析:")
            analyze_failure_reason(case, intent_result, debug_info)

def analyze_failure_reason(case, intent_result, debug_info):
    """分析失败原因"""
    user_input = case['input']
    expected = case['expected']
    actual = intent_result.get('intent', 'unknown')
    confidence = intent_result.get('confidence', 0.0)
    
    print(f"  1. 规则匹配问题:")
    
    # 检查是否是规则覆盖问题
    if actual == "unknown" and confidence == 0.0:
        print(f"     - 没有规则匹配到输入 '{user_input}'")
        print(f"     - 建议: 为{expected}意图添加相应的正则模式")
    
    # 检查是否是意图冲突问题
    elif actual != expected and confidence > 0.5:
        print(f"     - 规则匹配到了错误的意图: {actual}")
        print(f"     - 可能原因: {actual}意图的规则过于宽泛")
        print(f"     - 建议: 调整规则优先级或增加{expected}意图的特征词")
    
    # 检查是否是置信度阈值问题  
    elif confidence < 0.7:
        print(f"     - 置信度过低 ({confidence:.3f} < 0.7)")
        print(f"     - 触发了澄清机制")
        
        clarification_result = debug_info.get("clarification_result")
        if clarification_result:
            clarified_intent = clarification_result.get('intent')
            clarified_conf = clarification_result.get('confidence', 0)
            if clarified_intent == expected:
                print(f"     - LLM澄清成功识别为{expected} (conf:{clarified_conf:.3f})")
                print(f"     - 但最终输出仍显示为{actual}，可能是澄清结果未正确应用")
            else:
                print(f"     - LLM澄清也失败，识别为{clarified_intent}")
                print(f"     - 建议: 优化澄清提示词或增加更多上下文信息")
        else:
            print(f"     - 澄清机制未成功生成候选")
            print(f"     - 建议: 检查LLM提示词是否合适")
    
    print(f"  2. 实体抽取问题:")
    entities = intent_result.get('entities', [])
    if not entities:
        print(f"     - 未识别出任何实体")
        print(f"     - 可能影响意图分类的准确性")
    else:
        entity_types = [e.get('entity_type') for e in entities]
        print(f"     - 识别出实体类型: {entity_types}")
        
        # 检查是否缺少关键实体
        if expected == "query_status" and "device" not in entity_types:
            print(f"     - 查询意图缺少设备实体，可能影响分类")
        elif expected == "device_control" and "action" not in entity_types:
            print(f"     - 控制意图缺少动作实体，可能影响分类")

if __name__ == "__main__":
    debug_specific_cases()

