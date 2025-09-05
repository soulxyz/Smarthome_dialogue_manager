#!/usr/bin/env python3
"""语义覆盖度演示脚本

演示当前对话系统对各种语义表达的处理能力，包括：
1. 当前规则能处理的标准表达
2. 规则无法匹配但LLM可能恢复的变体表达
3. 系统完全无法理解的表达

用法:
    python scripts/demo_semantic_coverage.py [--api-key YOUR_API_KEY] [--interactive]
"""

import sys
import os
from pathlib import Path
import argparse
import time
from typing import Dict, List, Tuple, Any
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dialogue_manager.engine import DialogueEngine, EngineConfig, DialogueState
from tests.test_semantic_coverage_cases import SEMANTIC_TEST_CASES


class SemanticCoverageDemo:
    """语义覆盖度演示类"""
    
    def __init__(self, api_key: str):
        """初始化演示器"""
        self.config = EngineConfig(
            confidence_threshold=0.7,
            enable_clarification=True,
            enable_device_manager=True,
            max_turns=10
        )
        
        try:
            self.engine = DialogueEngine(api_key, self.config)
            self.api_available = True
        except Exception as e:
            print(f"⚠️  API初始化失败: {e}")
            print("将以离线模式运行，只测试规则匹配...\n")
            self.api_available = False
    
    def run_comprehensive_demo(self):
        """运行全面演示"""
        print("🔍 语义覆盖度演示 - 对话系统理解能力评估")
        print("=" * 60)
        print()
        
        # 收集所有测试结果
        all_results = {}
        
        # 1. 设备控制表达演示
        print("📱 设备控制表达演示")
        print("-" * 30)
        all_results.update(self._demo_device_control())
        
        # 2. 查询状态表达演示
        print("\n❓ 查询状态表达演示") 
        print("-" * 30)
        all_results.update(self._demo_query_status())
        
        # 3. 场景控制表达演示
        print("\n🎭 场景控制表达演示")
        print("-" * 30)
        all_results.update(self._demo_scene_control())
        
        # 4. 上下文相关表达演示
        if self.api_available:
            print("\n🔗 上下文相关表达演示")
            print("-" * 30)
            all_results.update(self._demo_contextual_expressions())
        
        # 5. 生成总结报告
        print("\n📊 演示总结")
        print("=" * 60)
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _demo_device_control(self) -> Dict[str, Any]:
        """演示设备控制表达"""
        results = {}
        
        # 标准表达 vs 变体表达对比
        comparisons = [
            {
                "category": "开启动作",
                "standard": "打开客厅灯",
                "variants": ["把客厅灯点亮", "让客厅灯亮起来", "客厅灯光亮起来吧"]
            },
            {
                "category": "调节动作", 
                "standard": "调高空调温度",
                "variants": ["空调温度升一下", "空调热一点", "温度调高些"]
            },
            {
                "category": "关闭动作",
                "standard": "关闭电视",
                "variants": ["把电视停掉", "让电视别响了", "电视关掉"]
            }
        ]
        
        for comp in comparisons:
            print(f"\n🎯 {comp['category']}对比:")
            
            # 测试标准表达
            std_result = self._test_expression(comp['standard'], "device_control")
            print(f"  标准: {comp['standard']:20} -> {self._format_result(std_result)} ✓")
            
            # 测试变体表达
            variant_results = []
            for variant in comp['variants']:
                var_result = self._test_expression(variant, "device_control")
                variant_results.append(var_result)
                status = self._get_result_status(var_result, "device_control")
                print(f"  变体: {variant:20} -> {self._format_result(var_result)} {status}")
            
            results[comp['category']] = {
                "standard": std_result,
                "variants": variant_results
            }
        
        return results
    
    def _demo_query_status(self) -> Dict[str, Any]:
        """演示查询状态表达"""
        results = {}
        
        query_cases = [
            {
                "category": "温度查询",
                "standard": "查看当前温度",
                "variants": ["温度多少?", "现在几度?", "热不热啊?"]
            },
            {
                "category": "设备状态查询",
                "standard": "检查空调状态", 
                "variants": ["空调开着没?", "空调怎么样?", "空调工作吗?"]
            }
        ]
        
        for case in query_cases:
            print(f"\n🔍 {case['category']}:")
            
            std_result = self._test_expression(case['standard'], "query_status")
            print(f"  标准: {case['standard']:20} -> {self._format_result(std_result)} ✓")
            
            variant_results = []
            for variant in case['variants']:
                var_result = self._test_expression(variant, "query_status")
                variant_results.append(var_result)
                status = self._get_result_status(var_result, "query_status")
                print(f"  变体: {variant:20} -> {self._format_result(var_result)} {status}")
            
            results[case['category']] = {
                "standard": std_result,
                "variants": variant_results
            }
        
        return results
    
    def _demo_scene_control(self) -> Dict[str, Any]:
        """演示场景控制表达"""
        results = {}
        
        scene_cases = [
            {
                "category": "睡眠场景",
                "standard": "启动睡眠模式",
                "variants": ["我要睡觉了", "准备休息", "要睡觉"]
            },
            {
                "category": "观影场景",
                "standard": "切换到观影模式",
                "variants": ["看个电影", "要看片子", "准备看电视"]
            }
        ]
        
        for case in scene_cases:
            print(f"\n🎬 {case['category']}:")
            
            std_result = self._test_expression(case['standard'], "scene_control")
            print(f"  标准: {case['standard']:20} -> {self._format_result(std_result)} ✓")
            
            variant_results = []
            for variant in case['variants']:
                var_result = self._test_expression(variant, "scene_control")
                variant_results.append(var_result)
                status = self._get_result_status(var_result, "scene_control")
                print(f"  变体: {variant:20} -> {self._format_result(var_result)} {status}")
            
            results[case['category']] = {
                "standard": std_result, 
                "variants": variant_results
            }
        
        return results
    
    def _demo_contextual_expressions(self) -> Dict[str, Any]:
        """演示上下文相关表达"""
        if not self.api_available:
            print("⚠️  需要API支持，跳过上下文测试")
            return {}
        
        print("建立上下文: 先提到一个设备...")
        
        # 重置会话状态（手动清理）
        self.engine.dialogue_history.clear()
        self.engine.context.clear()
        self.engine.current_state = DialogueState.LISTENING
        
        # 建立上下文
        context_setup = "客厅有台电视"
        setup_result = self._test_expression(context_setup, None, show_details=False)
        print(f"  上下文: {context_setup} -> {self._format_result(setup_result)}")
        
        # 测试上下文引用
        contextual_cases = [
            ("这个开一下", "device_control", "指代电视"),
            ("调大一点", "device_control", "调大电视音量"),
            ("这个怎么样?", "query_status", "查询电视状态"),
        ]
        
        results = {"contextual_cases": []}
        
        for expr, expected_intent, description in contextual_cases:
            result = self._test_expression(expr, expected_intent, show_details=False)
            results["contextual_cases"].append({
                "expression": expr,
                "result": result,
                "description": description
            })
            
            status = self._get_result_status(result, expected_intent)
            print(f"  引用: {expr:15} -> {self._format_result(result)} {status} ({description})")
        
        return results
    
    def _test_expression(self, expression: str, expected_intent: str = None, 
                        show_details: bool = False) -> Dict[str, Any]:
        """测试单个表达式"""
        try:
            if not self.api_available:
                # 只使用意图识别器进行规则匹配测试
                intent_result = self.engine.intent_recognizer.recognize(
                    expression, {}, []
                )
                return {
                    "expression": expression,
                    "intent": intent_result.get("intent", "unknown"),
                    "confidence": intent_result.get("confidence", 0.0),
                    "need_clarification": intent_result.get("need_clarification", False),
                    "api_used": False
                }
            
            # 完整引擎测试
            start_time = time.time()
            response, debug_info = self.engine.process_input(expression)
            processing_time = time.time() - start_time
            
            intent_result = debug_info.get("intent_result", {})
            
            result = {
                "expression": expression,
                "intent": intent_result.get("intent", "unknown"),
                "confidence": intent_result.get("confidence", 0.0),
                "need_clarification": intent_result.get("need_clarification", False),
                "response": response,
                "processing_time": processing_time,
                "api_used": True
            }
            
            # 检查澄清结果
            if "clarification_result" in debug_info:
                result["clarification_result"] = debug_info["clarification_result"]
            
            if show_details:
                print(f"    详细: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            return result
            
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "intent": "error",
                "confidence": 0.0
            }
    
    def _format_result(self, result: Dict[str, Any]) -> str:
        """格式化结果显示"""
        if "error" in result:
            return f"ERROR: {result['error']}"
        
        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.0)
        return f"{intent:15} (conf:{confidence:.2f})"
    
    def _get_result_status(self, result: Dict[str, Any], expected_intent: str = None) -> str:
        """获取结果状态符号"""
        if "error" in result:
            return "❌"
        
        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.0)
        need_clarification = result.get("need_clarification", False)
        
        if expected_intent and intent == expected_intent:
            if confidence >= self.config.confidence_threshold:
                return "✅"  # 规则直接匹配
            elif not need_clarification:
                return "🟡"  # 低置信度但无澄清
            else:
                return "🔄"  # 需要澄清
        elif expected_intent and intent != expected_intent:
            if need_clarification and self.api_available:
                return "⚠️"   # 可能通过澄清恢复
            else:
                return "❌"  # 完全失败
        else:
            return "❓"  # 无预期意图
    
    def _generate_summary_report(self, all_results: Dict[str, Any]):
        """生成总结报告"""
        total_cases = 0
        rule_matches = 0
        partial_matches = 0
        failures = 0
        
        for category, data in all_results.items():
            if isinstance(data, dict) and "variants" in data:
                # 标准格式的结果
                std_result = data["standard"]
                if std_result.get("confidence", 0) >= self.config.confidence_threshold:
                    rule_matches += 1
                total_cases += 1
                
                for var_result in data["variants"]:
                    total_cases += 1
                    conf = var_result.get("confidence", 0)
                    if conf >= self.config.confidence_threshold:
                        rule_matches += 1
                    elif conf > 0.3:  # 部分理解
                        partial_matches += 1
                    else:
                        failures += 1
            
            elif isinstance(data, dict) and "contextual_cases" in data:
                # 上下文测试结果
                for case in data["contextual_cases"]:
                    total_cases += 1
                    result = case["result"]
                    conf = result.get("confidence", 0)
                    if conf >= self.config.confidence_threshold:
                        rule_matches += 1
                    elif conf > 0.3:
                        partial_matches += 1
                    else:
                        failures += 1
        
        if total_cases == 0:
            print("未收集到测试数据")
            return
        
        print(f"📈 整体表现:")
        print(f"  测试用例总数: {total_cases}")
        print(f"  高置信度匹配: {rule_matches} ({rule_matches/total_cases:.1%}) ✅")
        print(f"  部分理解: {partial_matches} ({partial_matches/total_cases:.1%}) 🟡") 
        print(f"  完全失败: {failures} ({failures/total_cases:.1%}) ❌")
        print()
        
        # 给出改进建议
        coverage_rate = rule_matches / total_cases
        if coverage_rate >= 0.8:
            print("🎉 系统语义覆盖度表现优秀！")
        elif coverage_rate >= 0.6:
            print("👍 系统语义覆盖度表现良好，有改进空间。")
        elif coverage_rate >= 0.4:
            print("⚠️  系统语义覆盖度中等，建议重点改进规则匹配。")
        else:
            print("🚨 系统语义覆盖度较低，建议优先改进意图识别机制。")
        
        if not self.api_available:
            print("\n💡 建议配置API密钥以测试完整的LLM澄清恢复能力。")
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("🔍 交互式语义覆盖度测试")
        print("=" * 40)
        print("输入表达式来测试系统的理解能力")
        print("输入 'quit' 或 'exit' 退出\n")
        
        while True:
            try:
                user_input = input("👤 请输入测试表达: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break
                
                if not user_input:
                    continue
                
                # 重置会话状态
                if self.api_available:
                    self.engine.dialogue_history.clear()
                    self.engine.context.clear()
                
                print(f"🤖 正在处理: {user_input}")
                result = self._test_expression(user_input, show_details=True)
                
                print(f"   结果: {self._format_result(result)}")
                if result.get("need_clarification"):
                    print("   🔄 系统认为需要澄清")
                if result.get("response"):
                    print(f"   💬 系统回复: {result['response']}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 处理错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="语义覆盖度演示脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python scripts/demo_semantic_coverage.py
  python scripts/demo_semantic_coverage.py --api-key sk-xxx --interactive
  python scripts/demo_semantic_coverage.py --offline-only
        """
    )
    
    parser.add_argument("--api-key", 
                       help="API密钥 (也可以设置环境变量 API_KEY)")
    parser.add_argument("--interactive", action="store_true",
                       help="运行交互式模式")
    parser.add_argument("--offline-only", action="store_true", 
                       help="仅运行离线测试（不使用API）")
    
    args = parser.parse_args()
    
    # 获取API密钥 - 使用与测试系统相同的方式
    api_key = args.api_key or os.getenv("API_KEY") or os.getenv("DIALOGUE_API_KEY")
    
    # 如果仍然没有，尝试从测试配置获取
    if not api_key and not args.offline_only:
        try:
            from tests.config import get_test_config
            config = get_test_config()
            if config.api_key and config.api_key != "sk-test-key-for-offline-testing":
                api_key = config.api_key
                print("✅ 使用测试配置中的API密钥")
        except Exception as e:
            pass
    
    if args.offline_only:
        api_key = "offline-mode"  # 占位符
    elif not api_key:
        print("⚠️  未提供API密钥，将以离线模式运行")
        print("   可以通过以下方式提供API密钥:")
        print("   1. --api-key 参数")
        print("   2. API_KEY 环境变量") 
        print("   3. DIALOGUE_API_KEY 环境变量")
        print("   4. tests/config/.env 文件")
        api_key = "offline-mode"
    
    # 创建演示器
    demo = SemanticCoverageDemo(api_key)
    
    try:
        if args.interactive:
            # 交互式模式
            demo.run_interactive_demo()
        else:
            # 全面演示模式
            results = demo.run_comprehensive_demo()
            
            # 可选：保存结果到文件
            if "--save-results" in sys.argv:
                output_file = "semantic_coverage_results.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n💾 结果已保存到: {output_file}")
    
    except KeyboardInterrupt:
        print("\n👋 演示中断")
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
