"""语义覆盖度测试

测试当前基于规则的系统无法匹配，但语义上应该能理解的表达。
这些测试用于评估现有LLM澄清机制是否能弥补规则匹配的不足。

CI注意：这些测试目前可能失败，用于识别系统改进机会，不应阻塞构建。
"""

import pytest
import warnings
import json
import time
from typing import Dict, List, Any, Optional

from dialogue_manager.engine import DialogueEngine, EngineConfig
from dialogue_manager.intent import IntentRecognizer, IntentType
from tests.fixtures import test_config, dialogue_engine, intent_recognizer


# 自定义警告类别
class SemanticCoverageWarning(UserWarning):
    """语义覆盖度警告"""
    pass


# 测试数据：当前规则无法匹配的表达
SEMANTIC_TEST_CASES = {
    "device_control": {
        "alternative_actions": [
            # 替代的开启表达
            "把客厅灯点亮",  # 没有"打开"
            "让电视运行起来",  # 没有"启动" 
            "帮我弄亮卧室的灯",  # 没有标准动作词
            "客厅灯光亮起来吧",  # 间接表达
            "麻烦开一下空调",  # 礼貌用语
            
            # 替代的关闭表达
            "把灯熄灭",  # 没有"关闭"
            "让电视停下来",  # 没有"关"
            "灯光暗下去",  # 间接表达
            "空调别运行了",  # 否定表达
            
            # 替代的调节表达  
            "空调温度升一下",  # 没有"调高"
            "灯光暗一些",  # 没有"调低"
            "声音大一点",  # 没有"增加"
            "风扇慢一点",  # 没有"减少"
            "温度热一点",  # 口语化表达
        ],
        
        "word_order_variants": [
            # 语序变化
            "温度调到25度",  # vs "调到温度25度"
            "25度设置空调",  # 颠倒语序
            "客厅的灯给我开开",  # 不标准语序
            "亮度高一点调节",  # 语序变化
        ],
        
        "colloquial_expressions": [
            # 口语化表达
            "来个灯",  # 极简表达
            "整个空调",  # 方言式
            "弄个电视看看",  # 口语化
            "搞点音乐",  # 非正式用语
            "给点光",  # 简化表达
        ],
        
        "contextual_references": [
            # 需要上下文理解的表达
            "这个开一下",  # 需要设备上下文
            "也开开",  # 需要参照物
            "同样操作",  # 需要历史操作
            "反过来",  # 需要相反操作理解
        ]
    },
    
    "query_status": {
        "alternative_queries": [
            # 替代查询表达
            "温度多少?",  # 语序不对
            "现在热不热?",  # 没有"温度"关键词
            "房间什么情况?",  # 词汇不匹配
            "空调开着没?",  # 口语化
            "灯亮着吗?",  # 状态查询
            "现在几度啊?",  # 方言化
        ],
        
        "implicit_queries": [
            # 隐式查询
            "好热啊",  # 隐含温度查询
            "有点暗",  # 隐含亮度查询
            "太吵了",  # 隐含音量查询
            "空气不好",  # 隐含空气质量查询
        ]
    },
    
    "scene_control": {
        "alternative_scenes": [
            # 替代场景表达
            "我要休息了",  # vs "睡眠模式"
            "准备睡觉",  # vs "启动睡眠模式"
            "看个电影",  # vs "观影模式"
            "朋友要来了",  # vs "聚会模式"
            "出去一趟",  # vs "离家模式"
            "到家了",  # vs "回家模式"
        ]
    },
    
    "schedule_task": {
        "alternative_schedules": [
            # 替代定时表达
            "明早8点开空调",  # vs "定时"
            "睡觉前关灯",  # 相对时间
            "半小时后提醒我",  # 时间间隔不标准
            "每天早上自动开",  # 重复任务非标准
        ]
    }
}


class TestSemanticCoverageEvaluation:
    """语义覆盖度评估测试"""
    
    def setup_method(self):
        """设置方法"""
        self.results = {
            "rule_matches": 0,
            "llm_recoveries": 0, 
            "total_failures": 0,
            "confidence_scores": [],
            "clarification_triggered": 0,
            "details": []
        }
    
    @pytest.mark.semantic_coverage
    @pytest.mark.api_required
    def test_alternative_device_actions(self, dialogue_engine):
        """测试替代设备控制表达"""
        cases = SEMANTIC_TEST_CASES["device_control"]["alternative_actions"]
        self._evaluate_expressions(dialogue_engine, cases, "device_control", "替代动作表达")
    
    @pytest.mark.semantic_coverage  
    @pytest.mark.api_required
    def test_word_order_variants(self, dialogue_engine):
        """测试语序变化表达"""
        cases = SEMANTIC_TEST_CASES["device_control"]["word_order_variants"]
        self._evaluate_expressions(dialogue_engine, cases, "device_control", "语序变化表达")
    
    @pytest.mark.semantic_coverage
    @pytest.mark.api_required  
    def test_colloquial_expressions(self, dialogue_engine):
        """测试口语化表达"""
        cases = SEMANTIC_TEST_CASES["device_control"]["colloquial_expressions"]
        self._evaluate_expressions(dialogue_engine, cases, "device_control", "口语化表达")
    
    @pytest.mark.semantic_coverage
    @pytest.mark.api_required
    def test_alternative_queries(self, dialogue_engine):
        """测试替代查询表达"""
        cases = SEMANTIC_TEST_CASES["query_status"]["alternative_queries"] 
        self._evaluate_expressions(dialogue_engine, cases, "query_status", "替代查询表达")
    
    @pytest.mark.semantic_coverage
    @pytest.mark.api_required
    def test_implicit_queries(self, dialogue_engine):
        """测试隐式查询表达"""
        cases = SEMANTIC_TEST_CASES["query_status"]["implicit_queries"]
        self._evaluate_expressions(dialogue_engine, cases, "query_status", "隐式查询表达")
    
    @pytest.mark.semantic_coverage
    @pytest.mark.api_required
    def test_alternative_scene_expressions(self, dialogue_engine):
        """测试替代场景表达"""
        cases = SEMANTIC_TEST_CASES["scene_control"]["alternative_scenes"]
        self._evaluate_expressions(dialogue_engine, cases, "scene_control", "替代场景表达")
    
    @pytest.mark.semantic_coverage
    @pytest.mark.api_required
    def test_alternative_schedule_expressions(self, dialogue_engine):
        """测试替代定时表达"""
        cases = SEMANTIC_TEST_CASES["schedule_task"]["alternative_schedules"]
        self._evaluate_expressions(dialogue_engine, cases, "schedule_task", "替代定时表达")
    
    def _evaluate_expressions(self, dialogue_engine, expressions: List[str], 
                            expected_intent: str, category: str):
        """评估表达式理解能力"""
        print(f"\n=== 评估{category} ===")
        
        category_results = {
            "category": category,
            "total_cases": len(expressions),
            "rule_matches": 0,
            "llm_recoveries": 0,
            "failures": 0,
            "cases": []
        }
        
        for expr in expressions:
            result = self._test_single_expression(dialogue_engine, expr, expected_intent)
            category_results["cases"].append(result)
            
            if result["rule_matched"]:
                category_results["rule_matches"] += 1
            elif result["llm_recovered"]: 
                category_results["llm_recoveries"] += 1
            else:
                category_results["failures"] += 1
        
        # 生成测试报告
        self._generate_category_report(category_results)
        
        # 更新全局结果
        self.results["rule_matches"] += category_results["rule_matches"]
        self.results["llm_recoveries"] += category_results["llm_recoveries"] 
        self.results["total_failures"] += category_results["failures"]
        self.results["details"].append(category_results)
        
        # 生成警告而不是失败
        coverage_rate = (category_results["rule_matches"] + category_results["llm_recoveries"]) / category_results["total_cases"]
        
        if coverage_rate < 0.7:  # 覆盖率低于70%
            warning_msg = (f"{category}语义覆盖率较低: {coverage_rate:.1%} "
                         f"(规则匹配: {category_results['rule_matches']}, "
                         f"LLM恢复: {category_results['llm_recoveries']}, "
                         f"失败: {category_results['failures']})")
            warnings.warn(warning_msg, SemanticCoverageWarning)
    
    def _test_single_expression(self, dialogue_engine, expression: str, 
                               expected_intent: str) -> Dict[str, Any]:
        """测试单个表达式"""
        start_time = time.time()
        
        try:
            # 重置引擎状态
            dialogue_engine.dialogue_history.clear()
            dialogue_engine.context.clear()
            
            # 处理输入
            response, debug_info = dialogue_engine.process_input(expression)
            processing_time = time.time() - start_time
            
            # 分析结果
            intent_result = debug_info.get("intent_result", {})
            recognized_intent = intent_result.get("intent", "unknown")
            confidence = intent_result.get("confidence", 0.0)
            need_clarification = intent_result.get("need_clarification", False)
            
            # 判断是否规则直接匹配
            rule_matched = (recognized_intent == expected_intent and 
                          confidence >= dialogue_engine.config.confidence_threshold)
            
            # 判断是否通过LLM澄清机制恢复
            llm_recovered = False
            if need_clarification:
                # 检查是否有澄清候选或自澄清结果
                clarification_result = debug_info.get("clarification_result")
                if clarification_result:
                    # 如果澄清成功并得到正确意图，认为是LLM恢复
                    clarified_intent = clarification_result.get("intent")
                    clarified_confidence = clarification_result.get("confidence", 0.0)
                    if clarified_intent == expected_intent and clarified_confidence > confidence:
                        llm_recovered = True
            
            result = {
                "expression": expression,
                "expected_intent": expected_intent,
                "recognized_intent": recognized_intent,
                "confidence": confidence,
                "rule_matched": rule_matched,
                "llm_recovered": llm_recovered,
                "need_clarification": need_clarification,
                "processing_time": processing_time,
                "success": rule_matched or llm_recovered,
                "debug_info": debug_info
            }
            
            print(f"  {expression:30} -> {recognized_intent:15} (conf:{confidence:.2f}) "
                  f"{'✓规则' if rule_matched else '✓LLM' if llm_recovered else '✗失败'}")
            
            return result
            
        except Exception as e:
            print(f"  {expression:30} -> ERROR: {str(e)}")
            return {
                "expression": expression,
                "expected_intent": expected_intent,
                "error": str(e),
                "rule_matched": False,
                "llm_recovered": False,
                "success": False,
                "processing_time": time.time() - start_time
            }
    
    def _generate_category_report(self, category_results: Dict[str, Any]):
        """生成分类报告"""
        total = category_results["total_cases"]
        rule_matches = category_results["rule_matches"]
        llm_recoveries = category_results["llm_recoveries"]
        failures = category_results["failures"]
        
        print(f"\n{category_results['category']}结果总结:")
        print(f"  总测试用例: {total}")
        print(f"  规则直接匹配: {rule_matches} ({rule_matches/total:.1%})")
        print(f"  LLM澄清恢复: {llm_recoveries} ({llm_recoveries/total:.1%})")
        print(f"  完全失败: {failures} ({failures/total:.1%})")
        print(f"  综合覆盖率: {(rule_matches + llm_recoveries)/total:.1%}")
        
        # 详细失败分析
        if failures > 0:
            print(f"\n失败用例分析:")
            for case in category_results["cases"]:
                if not case["success"]:
                    expr = case["expression"]
                    intent = case.get("recognized_intent", "ERROR")
                    conf = case.get("confidence", 0)
                    print(f"  - '{expr}' -> {intent} (conf:{conf:.2f})")


class TestContextualSemanticCases:
    """上下文相关的语义测试"""
    
    @pytest.mark.semantic_coverage
    @pytest.mark.api_required
    def test_contextual_references(self, dialogue_engine):
        """测试需要上下文的引用表达"""
        # 设置上下文：先提到一个设备
        dialogue_engine.reset_session()
        
        print("\n=== 上下文引用测试 ===")
        
        # 建立上下文
        setup_response, setup_debug = dialogue_engine.process_input("客厅有台电视")
        print(f"建立上下文: 客厅有台电视 -> {setup_debug.get('intent_result', {}).get('intent', 'unknown')}")
        
        # 测试上下文引用
        contextual_cases = [
            ("这个开一下", "device_control"),  # 指代电视
            ("也关掉", "device_control"),      # 也关掉电视  
            ("调大一点", "device_control"),     # 调大电视音量
            ("这个怎么样?", "query_status"),    # 查询电视状态
        ]
        
        successes = 0
        for expr, expected_intent in contextual_cases:
            response, debug_info = dialogue_engine.process_input(expr)
            intent_result = debug_info.get("intent_result", {})
            recognized_intent = intent_result.get("intent", "unknown")
            confidence = intent_result.get("confidence", 0.0)
            
            success = recognized_intent == expected_intent or confidence >= 0.5
            if success:
                successes += 1
                
            print(f"  {expr:20} -> {recognized_intent:15} (conf:{confidence:.2f}) "
                  f"{'✓' if success else '✗'}")
        
        coverage = successes / len(contextual_cases)
        if coverage < 0.5:
            warnings.warn(f"上下文引用覆盖率较低: {coverage:.1%}", SemanticCoverageWarning)


class TestSemanticRobustness:
    """语义鲁棒性测试"""
    
    @pytest.mark.semantic_coverage
    @pytest.mark.api_required
    def test_typos_and_variations(self, dialogue_engine):
        """测试拼写错误和变体的鲁棒性"""
        typo_cases = [
            ("打开客听的灯", "device_control"),  # 客厅 -> 客听
            ("关比电视", "device_control"),      # 关掉 -> 关比  
            ("空调温度调到25度吧", "device_control"),  # 正常但口语化
            ("现再几度啊", "query_status"),      # 现在 -> 现再
        ]
        
        print("\n=== 拼写错误鲁棒性测试 ===")
        
        successes = 0
        for expr, expected_intent in typo_cases:
            try:
                dialogue_engine.reset_session()
                response, debug_info = dialogue_engine.process_input(expr)
                intent_result = debug_info.get("intent_result", {})
                recognized_intent = intent_result.get("intent", "unknown")
                confidence = intent_result.get("confidence", 0.0)
                
                # 对于拼写错误，我们降低成功标准
                success = (recognized_intent == expected_intent or 
                          confidence >= 0.3 or  # 较低置信度也接受
                          intent_result.get("need_clarification", False))  # 触发澄清也算成功
                
                if success:
                    successes += 1
                    
                print(f"  {expr:25} -> {recognized_intent:15} (conf:{confidence:.2f}) "
                      f"{'✓' if success else '✗'}")
                      
            except Exception as e:
                print(f"  {expr:25} -> ERROR: {str(e)}")
        
        coverage = successes / len(typo_cases)
        if coverage < 0.3:  # 拼写错误容忍度更低
            warnings.warn(f"拼写错误鲁棒性较低: {coverage:.1%}", SemanticCoverageWarning)


@pytest.fixture
def semantic_coverage_config():
    """语义覆盖度测试配置"""
    return EngineConfig(
        confidence_threshold=0.6,  # 稍微降低阈值
        enable_clarification=True,
        enable_device_manager=True,
        max_turns=20
    )


def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "semantic_coverage: 语义覆盖度测试，可能失败但不阻塞CI"
    )


def pytest_runtest_makereport(item, call):
    """自定义测试报告生成"""
    if "semantic_coverage" in item.keywords:
        # 对于语义覆盖度测试，将失败转换为警告
        if call.when == "call" and call.excinfo:
            # 如果是我们的语义覆盖度警告，不算作测试失败
            if isinstance(call.excinfo.value, SemanticCoverageWarning):
                call.excinfo = None  # 清除异常信息
                
    return None
