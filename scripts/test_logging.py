#!/usr/bin/env python3
"""日志系统测试脚本

演示如何使用新的日志功能并验证其正常工作。
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from dialogue_manager.logger import get_dialogue_logger, EventType, LogLevel


def test_basic_logging():
    """测试基本日志功能"""
    print("🧪 测试基本日志记录功能...")
    
    logger = get_dialogue_logger()
    
    # 模拟会话开始
    session_id = f"test_session_{int(time.time())}"
    user_id = "test_user"
    
    logger.log_session_event(
        session_id=session_id,
        event_type=EventType.SESSION_START,
        user_id=user_id
    )
    print(f"✅ 记录会话开始: {session_id}")
    
    # 模拟对话轮次
    for turn_id in range(1, 4):
        debug_info = {
            "intent_result": {
                "intent": "device_control" if turn_id % 2 == 1 else "query_status",
                "confidence": 0.85 + (turn_id * 0.05),
                "need_clarification": False
            },
            "processing_time": 1.2 + (turn_id * 0.3),
            "api_calls": [
                {
                    "success": True,
                    "response_time": 0.8,
                    "content": f"模拟响应内容 {turn_id}"
                }
            ],
            "context_updates": {
                "current_focus": {"value": "灯", "turn_count": 0}
            }
        }
        
        logger.log_dialogue_turn(
            session_id=session_id,
            turn_id=turn_id,
            user_input=f"测试用户输入 {turn_id}",
            system_response=f"测试系统响应 {turn_id}",
            debug_info=debug_info,
            user_id=user_id
        )
        print(f"✅ 记录对话轮次 {turn_id}")
    
    # 模拟API调用
    api_call_info = {
        "success": True,
        "response_time": 1.5,
        "content": "API调用成功",
        "request": {"model": "deepseek-chat", "messages": []},
        "response": {"usage": {"total_tokens": 150}}
    }
    
    logger.log_api_call(
        session_id=session_id,
        api_call_info=api_call_info,
        user_id=user_id,
        turn_id=2
    )
    print("✅ 记录API调用")
    
    # 模拟焦点切换
    logger.log_focus_switch(
        session_id=session_id,
        old_focus="灯",
        new_focus="空调",
        reason="用户明确提及新设备",
        user_id=user_id,
        turn_id=3
    )
    print("✅ 记录焦点切换")
    
    # 模拟错误
    try:
        raise ValueError("这是一个测试错误")
    except Exception as e:
        logger.log_error(
            session_id=session_id,
            error=e,
            context={"operation": "test", "step": "error_simulation"},
            user_id=user_id,
            turn_id=3
        )
    print("✅ 记录错误日志")
    
    # 模拟会话结束
    logger.log_session_event(
        session_id=session_id,
        event_type=EventType.SESSION_END,
        user_id=user_id,
        context={"total_turns": 3, "session_duration": 45.6}
    )
    print(f"✅ 记录会话结束: {session_id}")
    
    return session_id


def test_log_search():
    """测试日志搜索功能"""
    print("\n🔍 测试日志搜索功能...")
    
    logger = get_dialogue_logger()
    
    # 搜索最近的对话轮次
    dialogue_logs = logger.search_logs(
        event_type=EventType.DIALOGUE_TURN.value,
        limit=5
    )
    print(f"✅ 找到 {len(dialogue_logs)} 条对话轮次日志")
    
    # 搜索错误日志
    error_logs = logger.search_logs(
        level=LogLevel.ERROR.value,
        limit=5
    )
    print(f"✅ 找到 {len(error_logs)} 条错误日志")
    
    # 按关键词搜索
    keyword_logs = logger.search_logs(
        keyword="测试",
        limit=10
    )
    print(f"✅ 关键词搜索找到 {len(keyword_logs)} 条日志")


def test_session_summary(session_id):
    """测试会话摘要功能"""
    print(f"\n📊 测试会话摘要功能: {session_id}")
    
    logger = get_dialogue_logger()
    
    try:
        summary = logger.get_session_summary(session_id)
        
        if "error" not in summary:
            print("✅ 会话摘要生成成功:")
            print(f"  - 总轮数: {summary['total_turns']}")
            print(f"  - 错误数: {summary['total_errors']}")
            print(f"  - 平均处理时间: {summary['avg_processing_time']:.2f}s")
            print(f"  - API调用总数: {summary['total_api_calls']}")
            print(f"  - 意图分布: {summary['intent_distribution']}")
        else:
            print(f"❌ 获取会话摘要失败: {summary['error']}")
    except Exception as e:
        print(f"❌ 会话摘要测试出错: {e}")


def test_performance_metrics():
    """测试性能指标记录"""
    print("\n⚡ 测试性能指标记录...")
    
    logger = get_dialogue_logger()
    
    session_id = f"perf_test_{int(time.time())}"
    
    # 记录几个性能指标
    metrics = [
        ("intent_recognition_time", 0.15, "seconds"),
        ("device_response_time", 0.3, "seconds"),
        ("memory_usage", 125.5, "MB"),
        ("cpu_usage", 45.2, "percent")
    ]
    
    for metric_name, value, unit in metrics:
        logger.save_performance_metric(
            session_id=session_id,
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            context={"test": True, "timestamp": time.time()}
        )
        print(f"✅ 记录性能指标: {metric_name} = {value} {unit}")


def test_log_export():
    """测试日志导出功能"""
    print("\n📥 测试日志导出功能...")
    
    logger = get_dialogue_logger()
    
    try:
        import tempfile
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_file = f.name
        
        # 导出最近的日志
        logger.export_logs(
            output_file=temp_file,
            start_time=time.time() - 3600,  # 最近1小时
            end_time=time.time()
        )
        
        # 检查文件内容
        with open(temp_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"✅ 日志导出成功: {len(lines)} 条记录")
        print(f"   导出文件: {temp_file}")
        
        # 显示前两条记录的示例
        if lines:
            print("   示例记录:")
            for i, line in enumerate(lines[:2]):
                try:
                    import json
                    log_data = json.loads(line)
                    print(f"     {i+1}. {log_data.get('event_type')} - {log_data.get('message', '')[:50]}...")
                except:
                    print(f"     {i+1}. 无法解析的记录")
        
        # 清理临时文件
        import os
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"❌ 日志导出测试失败: {e}")


def test_cleanup():
    """测试日志清理功能"""
    print("\n🗑️ 测试日志清理功能...")
    
    logger = get_dialogue_logger()
    
    try:
        # 模拟清理7天前的日志
        logger.cleanup_old_logs(days_to_keep=7)
        print("✅ 日志清理执行成功")
    except Exception as e:
        print(f"❌ 日志清理测试失败: {e}")


def main():
    """主测试函数"""
    print("🚀 开始测试日志系统功能")
    print("=" * 50)
    
    try:
        # 执行各项测试
        session_id = test_basic_logging()
        test_log_search()
        test_session_summary(session_id)
        test_performance_metrics()
        test_log_export()
        test_cleanup()
        
        print("\n🎉 所有测试完成!")
        print("=" * 50)
        print("💡 提示:")
        print("1. 在UI界面中查看日志: streamlit run ui/app.py")
        print("2. 使用命令行工具: python scripts/debug_logs.py --help")
        print("3. 查看详细文档: docs/LOGGING_GUIDE.md")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
