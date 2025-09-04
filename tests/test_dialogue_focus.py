import unittest
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dialogue_manager.engine import DialogueEngine, EngineConfig

class TestDialogueFocus(unittest.TestCase):

    def setUp(self):
        """在每个测试用例前执行"""
        config = EngineConfig(enable_device_manager=True)
        self.engine = DialogueEngine(api_key="sk-fsjrtevskkmicnqdjjqdjarwqktxzwchkknnjmwgicczbubp", config=config)
        self.engine.start_session("test_user")

    def tearDown(self):
        """结束会话"""
        self.engine.end_session()

    def test_1_focus_tracking(self):
        """测试基本的焦点实体跟踪"""
        # 第一轮：建立焦点
        response, debug_info = self.engine.process_input("打开客厅的灯")
        self.assertIn("current_focus", self.engine.context)
        self.assertEqual(self.engine.context["current_focus"]["value"], "灯")
        self.assertEqual(self.engine.context["current_focus"]["turn_count"], 0)
        print(f"\nTest 1.1 Passed: {debug_info}")

        # 第二轮：焦点应该保持
        response, debug_info = self.engine.process_input("亮度调到50%")
        self.assertIn("current_focus", self.engine.context)
        self.assertEqual(self.engine.context["current_focus"]["value"], "灯")
        print(f"Test 1.2 Passed: {debug_info}")


    def test_2_pronoun_resolution(self):
        """测试代词消解"""
        # 第一轮：建立焦点
        self.engine.process_input("打开主卧的空调")
        self.assertIn("current_focus", self.engine.context)
        self.assertEqual(self.engine.context["current_focus"]["value"], "空调")
        print("\nTest 2.1 Passed: Focus set to '空调'")

        # 第二轮：使用代词“它”
        response, debug_info = self.engine.process_input("把它的温度调到26度")
        entities = debug_info.get("intent_result", {}).get("entities", [])
        device_entity = next((e for e in entities if e.get("entity_type") == "device"), None)
        self.assertIsNotNone(device_entity, "未能解析出设备实体")
        self.assertEqual(device_entity["value"], "空调", "代词未能正确解析为'空调'")
        print(f"Test 2.2 Passed: {debug_info}")

    def test_3_omitted_subject(self):
        """测试省略主语的指令"""
        # 第一轮：建立焦点
        self.engine.process_input("启动客厅的电视")
        self.assertIn("current_focus", self.engine.context)
        self.assertEqual(self.engine.context["current_focus"]["value"], "电视")
        print("\nTest 3.1 Passed: Focus set to '电视'")

        # 第二轮：省略主语
        response, debug_info = self.engine.process_input("关掉")
        entities = debug_info.get("intent_result", {}).get("entities", [])
        device_entity = next((e for e in entities if e.get("entity_type") == "device"), None)
        self.assertIsNotNone(device_entity, "未能从上下文中继承设备实体")
        self.assertEqual(device_entity["value"], "电视", "省略主语的指令未能正确解析")
        print(f"Test 3.2 Passed: {debug_info}")

    def test_4_focus_decay(self):
        """测试焦点实体的生命周期衰减"""
        # 默认 focus_entity_turn_decay = 2
        # 第一轮：建立焦点
        self.engine.process_input("打开书房的台灯")
        self.assertIn("current_focus", self.engine.context)
        print("\nTest 4.1 Passed: Focus set to '台灯'")

        # 第二轮：无关指令，turn_count 增加
        self.engine.process_input("今天天气怎么样")
        self.assertIn("current_focus", self.engine.context)
        self.assertEqual(self.engine.context["current_focus"]["turn_count"], 1)
        print("Test 4.2 Passed: Focus turn_count is 1")

        # 第三轮：无关指令，turn_count 增加并触发衰减
        self.engine.process_input("明天呢")
        self.assertNotIn("current_focus", self.engine.context, "焦点实体未按预期衰减移除")
        print("Test 4.3 Passed: Focus has decayed")

    def test_5_focus_switching(self):
        """测试焦点切换"""
        # 第一轮：建立焦点 "灯"
        self.engine.process_input("打开客厅的灯")
        self.assertIn("current_focus", self.engine.context)
        self.assertEqual(self.engine.context["current_focus"]["value"], "灯")
        print("\nTest 5.1 Passed: Focus set to '灯'")

        # 第二轮：提及新设备，切换焦点
        self.engine.process_input("打开厨房的空调")
        self.assertIn("current_focus", self.engine.context)
        self.assertEqual(self.engine.context["current_focus"]["value"], "空调")
        self.assertEqual(self.engine.context["current_focus"]["turn_count"], 0)
        print("Test 5.2 Passed: Focus switched to '空调'")

if __name__ == '__main__':
    unittest.main()