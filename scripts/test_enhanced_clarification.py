#!/usr/bin/env python3
"""测试增强后的澄清机制"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dialogue_manager.engine import DialogueEngine, EngineConfig
from tests.config import get_test_config

def test_enhanced_clarification():
    """测试增强后的澄清机制"""
    
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
    
    print("🔍 测试增强后的澄清机制")
    print("=" * 50)
    
    # 测试案例：需要设备管理器信息的模糊表达
    test_cases = [
        {
            "input": "开一下",
            "description": "完全模糊的开启指令，需要结合设备列表"
        },
        {
            "input": "调亮点",
            "description": "模糊的调节指令，需要推断设备"
        },
        {
            "input": "那个怎么样?",
            "description": "模糊查询，需要焦点设备信息"
        },
        {
            "input": "来个电影",
            "description": "场景控制，需要相关设备联动"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 测试案例 {i}: {case['description']}")
        print(f"输入: '{case['input']}'")
        print("-" * 30)
        
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
        
        print(f"识别意图: {recognized_intent}")
        print(f"置信度: {confidence:.3f}")
        print(f"需要澄清: {need_clarification}")
        
        # 检查澄清过程
        clarification_result = debug_info.get("clarification_result")
        if clarification_result:
            print(f"澄清结果: {clarification_result.get('intent')} (conf:{clarification_result.get('confidence', 0):.3f})")
            print(f"澄清文本: {clarification_result.get('clarified_text', 'N/A')}")
        
        clarification_candidates = debug_info.get("clarification_candidates", [])
        if clarification_candidates:
            print(f"澄清候选: {clarification_candidates}")
        
        print(f"系统回复: {response}")
        
        # 测试上下文继承 - 建立焦点后再测试
        if i == 1:  # 第一个案例后建立上下文
            print("\n🔄 建立上下文: 先操作一个设备...")
            engine.process_input("打开客厅灯")
            print("已建立焦点设备: 客厅灯")

if __name__ == "__main__":
    test_enhanced_clarification()

