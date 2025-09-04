"""省略消解和焦点切换功能测试"""

import unittest
from dialogue_manager.engine import DialogueEngine, EngineConfig


class TestOmissionResolution(unittest.TestCase):
    """测试省略消解和焦点切换功能"""

    def setUp(self):
        """测试前准备"""
        config = EngineConfig(
            enable_device_manager=True,
            enable_context_entity_fill=True,
            focus_switch_policy="conservative"
        )
        self.engine = DialogueEngine(api_key="test_key", config=config)
        self.engine.start_session("test_user")

    def tearDown(self):
        """测试后清理"""
        self.engine.end_session()

    def test_omission_resolution(self):
        """测试省略主语的指令处理"""
        # 首先建立焦点
        response, debug_info = self.engine.process_input("打开客厅的灯")
        self.assertIsNotNone(response)
        self.assertEqual(self.engine.context["current_focus"]["value"], "灯")

        # 测试省略主语的指令
        response, debug_info = self.engine.process_input("关掉")
        self.assertIsNotNone(response)
        
        # 验证实体抽取结果
        entities = debug_info["intent_result"]["entities"]
        device_entity = next((e for e in entities if e["entity_type"] == "device"), None)
        action_entity = next((e for e in entities if e["entity_type"] == "action"), None)
        
        self.assertIsNotNone(device_entity, "应该从上下文补充设备实体")
        self.assertEqual(device_entity["value"], "灯", "补充的设备实体应该是'灯'")
        self.assertIsNotNone(action_entity, "应该识别出动作实体")
        self.assertEqual(action_entity["value"], "关闭", "动作实体应该是'关闭'")

    def test_focus_switching(self):
        """测试焦点切换功能"""
        # 首先建立焦点
        response, debug_info = self.engine.process_input("打开客厅的灯")
        self.assertIsNotNone(response)
        self.assertEqual(self.engine.context["current_focus"]["value"], "灯")

        # 测试查询意图不切换焦点
        response, debug_info = self.engine.process_input("空调状态怎么样")
        self.assertIsNotNone(response)
        self.assertEqual(self.engine.context["current_focus"]["value"], "灯", "查询意图不应切换焦点")

        # 测试设备控制意图切换焦点
        response, debug_info = self.engine.process_input("打开空调")
        self.assertIsNotNone(response)
        self.assertEqual(self.engine.context["current_focus"]["value"], "空调", "设备控制意图应切换焦点")

        # 测试相同设备不切换焦点
        response, debug_info = self.engine.process_input("调高空调温度")
        self.assertIsNotNone(response)
        self.assertEqual(self.engine.context["current_focus"]["value"], "空调", "相同设备不应切换焦点")


if __name__ == "__main__":
    unittest.main()