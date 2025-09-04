"""焦点切换策略测试"""

import unittest
from dialogue_manager.engine import DialogueEngine, EngineConfig


class TestFocusSwitchingStrategies(unittest.TestCase):
    """测试不同的焦点切换策略"""

    def test_conservative_policy(self):
        """测试保守焦点切换策略"""
        config = EngineConfig(
            enable_device_manager=True,
            focus_switch_policy="conservative"
        )
        engine = DialogueEngine(api_key="test_key", config=config)
        engine.start_session("test_user")

        # 首先建立焦点
        engine.process_input("打开客厅的灯")
        self.assertEqual(engine.context["current_focus"]["value"], "灯")

        # 测试查询意图不切换焦点
        engine.process_input("空调状态怎么样")
        self.assertEqual(engine.context["current_focus"]["value"], "灯", "保守策略下查询意图不应切换焦点")

        # 测试设备控制意图切换焦点
        engine.process_input("打开空调")
        self.assertEqual(engine.context["current_focus"]["value"], "空调", "保守策略下设备控制意图应切换焦点")

        engine.end_session()

    def test_aggressive_policy(self):
        """测试激进焦点切换策略"""
        config = EngineConfig(
            enable_device_manager=True,
            focus_switch_policy="aggressive"
        )
        engine = DialogueEngine(api_key="test_key", config=config)
        engine.start_session("test_user")

        # 首先建立焦点
        engine.process_input("打开客厅的灯")
        self.assertEqual(engine.context["current_focus"]["value"], "灯")

        # 测试查询意图也切换焦点
        engine.process_input("空调状态怎么样")
        self.assertEqual(engine.context["current_focus"]["value"], "空调", "激进策略下查询意图也应切换焦点")

        engine.end_session()

    def test_disable_context_entity_fill(self):
        """测试禁用上下文实体填充"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=False
        )
        engine = DialogueEngine(api_key="test_key", config=config)
        engine.start_session("test_user")

        # 首先建立焦点
        engine.process_input("打开客厅的灯")
        self.assertEqual(engine.context["current_focus"]["value"], "灯")

        # 测试省略主语的指令
        response, debug_info = engine.process_input("关掉")
        
        # 验证实体抽取结果
        entities = debug_info["intent_result"]["entities"]
        device_entities = [e for e in entities if e["entity_type"] == "device"]
        
        # 禁用上下文实体填充时，不应该从上下文补充设备实体
        self.assertTrue(len(device_entities) == 0 or 
                      (len(device_entities) > 0 and device_entities[0]["confidence"] < 0.8),
                      "禁用上下文实体填充时不应从上下文补充设备实体")

        engine.end_session()


if __name__ == "__main__":
    unittest.main()