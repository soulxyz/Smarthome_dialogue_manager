"""P0核心可用性边界条件测试"""

import unittest
from dialogue_manager.engine import DialogueEngine, EngineConfig


class TestEdgeCasesP0(unittest.TestCase):
    """测试P0核心可用性边界条件"""

    def setUp(self):
        """测试前准备"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True,
            focus_switch_policy="conservative",
            focus_entity_turn_decay=2  # 焦点实体在2轮不被引用后失效
        )
        self.engine = DialogueEngine(api_key="test_key", config=config)
        self.engine.start_session("test_user")

    def tearDown(self):
        """测试后清理"""
        self.engine.end_session()

    def test_no_focus_entity(self):
        """测试没有焦点实体的情况"""
        # 没有建立焦点的情况下，省略主语的指令
        response, debug_info = self.engine.process_input("关掉")
        self.assertIsNotNone(response)
        
        # 验证实体抽取结果
        entities = debug_info["intent_result"]["entities"]
        device_entities = [e for e in entities if e["entity_type"] == "device"]
        
        # 没有焦点实体时，不应该从上下文补充设备实体
        self.assertEqual(len(device_entities), 0, "没有焦点实体时不应从上下文补充设备实体")

    def test_focus_entity_expiration(self):
        """测试焦点实体过期"""
        # 首先建立焦点
        self.engine.process_input("打开客厅的灯")
        self.assertEqual(self.engine.context["current_focus"]["value"], "灯")

        # 连续两轮不涉及焦点实体的对话
        self.engine.process_input("今天天气怎么样")
        self.engine.process_input("你是谁")
        
        # 第三轮，焦点应该已经过期
        self.assertNotIn("current_focus", self.engine.context, "焦点实体应该已经过期")

        # 此时省略主语的指令应该无法识别设备
        response, debug_info = self.engine.process_input("关掉")
        entities = debug_info["intent_result"]["entities"]
        device_entities = [e for e in entities if e["entity_type"] == "device"]
        
        self.assertEqual(len(device_entities), 0, "焦点过期后不应从上下文补充设备实体")

    def test_action_without_device(self):
        """测试只有动作没有设备的情况"""
        # 首先建立焦点
        self.engine.process_input("打开客厅的灯")
        self.assertEqual(self.engine.context["current_focus"]["value"], "灯")

        # 只有动作词的输入
        response, debug_info = self.engine.process_input("关闭")
        
        # 验证实体抽取结果
        entities = debug_info["intent_result"]["entities"]
        device_entity = next((e for e in entities if e["entity_type"] == "device"), None)
        action_entity = next((e for e in entities if e["entity_type"] == "action"), None)
        
        self.assertIsNotNone(device_entity, "应该从上下文补充设备实体")
        self.assertEqual(device_entity["value"], "灯", "补充的设备实体应该是'灯'")
        self.assertIsNotNone(action_entity, "应该识别出动作实体")
        self.assertEqual(action_entity["value"], "关闭", "动作实体应该是'关闭'")

    def test_device_without_action(self):
        """测试只有设备没有动作的情况"""
        # 只有设备名的输入
        response, debug_info = self.engine.process_input("空调")
        
        # 验证实体抽取结果
        entities = debug_info["intent_result"]["entities"]
        device_entity = next((e for e in entities if e["entity_type"] == "device"), None)
        
        self.assertIsNotNone(device_entity, "应该识别出设备实体")
        self.assertEqual(device_entity["value"], "空调", "设备实体应该是'空调'")
        
        # 验证焦点是否更新
        self.assertIn("current_focus", self.engine.context, "应该更新焦点实体")
        self.assertEqual(self.engine.context["current_focus"]["value"], "空调", "焦点实体应该是'空调'")


if __name__ == "__main__":
    unittest.main()