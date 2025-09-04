#!/usr/bin/env python3
"""日志调试和问题排查脚本

提供命令行工具用于分析对话日志、定位问题和生成报告。
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from dialogue_manager.logger import get_dialogue_logger, EventType, LogLevel


def format_timestamp(timestamp: float) -> str:
    """格式化时间戳"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def analyze_session(session_id: str):
    """分析特定会话的详细信息"""
    logger = get_dialogue_logger()
    
    print(f"\n🔍 分析会话: {session_id}")
    print("=" * 60)
    
    # 获取会话摘要
    try:
        summary = logger.get_session_summary(session_id)
        if "error" in summary:
            print(f"❌ 错误: {summary['error']}")
            return
        
        print("📊 会话摘要:")
        print(f"  - 总轮数: {summary['total_turns']}")
        print(f"  - 错误数: {summary['total_errors']}")
        print(f"  - 平均处理时间: {summary['avg_processing_time']:.2f}s")
        print(f"  - API调用总数: {summary['total_api_calls']}")
        print(f"  - 开始时间: {summary['start_time']}")
        print(f"  - 结束时间: {summary['end_time']}")
        print(f"  - 持续时间: {summary['duration_seconds']:.0f}秒")
        
        if summary['intent_distribution']:
            print(f"  - 意图分布: {summary['intent_distribution']}")
    
    except Exception as e:
        print(f"❌ 获取会话摘要失败: {e}")
        return
    
    # 获取详细日志
    logs = logger.search_logs(session_id=session_id, limit=1000)
    
    print(f"\n📝 详细日志 ({len(logs)} 条):")
    print("-" * 60)
    
    for log in logs:
        timestamp_str = format_timestamp(log.timestamp)
        print(f"[{timestamp_str}] {log.level:8s} | {log.event_type:15s} | {log.message}")
        
        if log.error_type:
            print(f"  ❌ 错误类型: {log.error_type}")
        
        if log.processing_time:
            print(f"  ⏱️  处理时间: {log.processing_time:.2f}s")
        
        if log.confidence:
            print(f"  🎯 置信度: {log.confidence:.2f}")
        
        print()


def find_errors(hours_back: int = 24, limit: int = 50):
    """查找最近的错误日志"""
    logger = get_dialogue_logger()
    
    start_time = time.time() - (hours_back * 3600)
    
    print(f"\n🚨 查找最近 {hours_back} 小时的错误日志")
    print("=" * 60)
    
    error_logs = logger.search_logs(
        level=LogLevel.ERROR.value,
        start_time=start_time,
        limit=limit
    )
    
    if not error_logs:
        print("✅ 没有发现错误日志")
        return
    
    print(f"发现 {len(error_logs)} 条错误日志:")
    print("-" * 60)
    
    # 按错误类型分组
    error_groups = {}
    for log in error_logs:
        error_type = log.error_type or "未知错误"
        if error_type not in error_groups:
            error_groups[error_type] = []
        error_groups[error_type].append(log)
    
    for error_type, logs in error_groups.items():
        print(f"\n📍 {error_type} ({len(logs)} 次):")
        for log in logs[:5]:  # 只显示前5个
            timestamp_str = format_timestamp(log.timestamp)
            print(f"  [{timestamp_str}] 会话:{log.session_id[-8:]} | {log.message}")
        
        if len(logs) > 5:
            print(f"  ... 还有 {len(logs) - 5} 条类似错误")


def analyze_performance(hours_back: int = 24):
    """分析性能指标"""
    logger = get_dialogue_logger()
    
    start_time = time.time() - (hours_back * 3600)
    
    print(f"\n⚡ 分析最近 {hours_back} 小时的性能指标")
    print("=" * 60)
    
    # 获取对话轮次日志
    dialogue_logs = logger.search_logs(
        event_type=EventType.DIALOGUE_TURN.value,
        start_time=start_time,
        limit=1000
    )
    
    # 获取API调用日志
    api_logs = logger.search_logs(
        event_type=EventType.API_CALL.value,
        start_time=start_time,
        limit=1000
    )
    
    if dialogue_logs:
        # 处理时间分析
        processing_times = [log.processing_time for log in dialogue_logs if log.processing_time]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            max_time = max(processing_times)
            min_time = min(processing_times)
            
            print(f"🎯 对话处理性能:")
            print(f"  - 总对话轮数: {len(dialogue_logs)}")
            print(f"  - 平均处理时间: {avg_time:.2f}s")
            print(f"  - 最长处理时间: {max_time:.2f}s")
            print(f"  - 最短处理时间: {min_time:.2f}s")
            
            # 慢请求分析
            slow_logs = [log for log in dialogue_logs if log.processing_time and log.processing_time > 5.0]
            if slow_logs:
                print(f"  - 慢请求数量: {len(slow_logs)} (>5s)")
                print("  - 慢请求详情:")
                for log in slow_logs[:3]:
                    timestamp_str = format_timestamp(log.timestamp)
                    print(f"    [{timestamp_str}] {log.processing_time:.2f}s | 会话:{log.session_id[-8:]}")
        
        # 意图识别分析
        intents = [log.intent for log in dialogue_logs if log.intent]
        if intents:
            intent_counts = {}
            for intent in intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            print(f"\n🎯 意图分布:")
            for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {intent}: {count} 次")
        
        # 置信度分析
        confidences = [log.confidence for log in dialogue_logs if log.confidence]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            low_confidence_count = len([c for c in confidences if c < 0.7])
            
            print(f"\n🎯 意图识别置信度:")
            print(f"  - 平均置信度: {avg_confidence:.3f}")
            print(f"  - 低置信度比例: {low_confidence_count}/{len(confidences)} ({low_confidence_count/len(confidences)*100:.1f}%)")
    
    # API调用分析
    if api_logs:
        successful_apis = [log for log in api_logs if log.context_data and log.context_data.get("success")]
        failed_apis = [log for log in api_logs if log.context_data and not log.context_data.get("success")]
        
        print(f"\n🌐 API调用性能:")
        print(f"  - 总API调用: {len(api_logs)}")
        print(f"  - 成功调用: {len(successful_apis)}")
        print(f"  - 失败调用: {len(failed_apis)}")
        
        if len(api_logs) > 0:
            success_rate = len(successful_apis) / len(api_logs) * 100
            print(f"  - 成功率: {success_rate:.1f}%")
        
        # API响应时间分析
        api_times = [log.processing_time for log in api_logs if log.processing_time]
        if api_times:
            avg_api_time = sum(api_times) / len(api_times)
            max_api_time = max(api_times)
            print(f"  - 平均API响应时间: {avg_api_time:.2f}s")
            print(f"  - 最长API响应时间: {max_api_time:.2f}s")


def export_session_report(session_id: str, output_file: str = None):
    """导出会话报告"""
    logger = get_dialogue_logger()
    
    if not output_file:
        output_file = f"session_report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    print(f"\n📄 导出会话报告: {session_id} -> {output_file}")
    print("=" * 60)
    
    try:
        # 获取会话摘要
        summary = logger.get_session_summary(session_id)
        
        # 获取详细日志
        logs = logger.search_logs(session_id=session_id, limit=1000)
        
        # 构建报告
        report = {
            "session_id": session_id,
            "export_time": datetime.now().isoformat(),
            "summary": summary,
            "logs": [log.to_dict() for log in logs]
        }
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 报告已保存到: {output_file}")
        print(f"  - 包含 {len(logs)} 条日志记录")
        
    except Exception as e:
        print(f"❌ 导出报告失败: {e}")


def monitor_realtime(refresh_interval: int = 30):
    """实时监控日志"""
    logger = get_dialogue_logger()
    
    print(f"\n👀 实时监控日志 (每 {refresh_interval} 秒刷新)")
    print("按 Ctrl+C 退出监控")
    print("=" * 60)
    
    last_check_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # 获取新日志
            new_logs = logger.search_logs(
                start_time=last_check_time,
                limit=100
            )
            
            if new_logs:
                print(f"\n[{format_timestamp(current_time)}] 发现 {len(new_logs)} 条新日志:")
                print("-" * 40)
                
                for log in new_logs:
                    timestamp_str = format_timestamp(log.timestamp)
                    level_symbol = "❌" if log.level == LogLevel.ERROR.value else "ℹ️"
                    print(f"{level_symbol} [{timestamp_str}] {log.event_type} | {log.message[:60]}...")
                    
                    if log.error_type:
                        print(f"  🚨 错误: {log.error_type}")
            
            last_check_time = current_time
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n👋 监控已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="对话日志调试和问题排查工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 分析会话命令
    session_parser = subparsers.add_parser('session', help='分析特定会话')
    session_parser.add_argument('session_id', help='会话ID')
    
    # 查找错误命令
    error_parser = subparsers.add_parser('errors', help='查找错误日志')
    error_parser.add_argument('--hours', type=int, default=24, help='查找最近N小时的错误 (默认: 24)')
    error_parser.add_argument('--limit', type=int, default=50, help='最大返回数量 (默认: 50)')
    
    # 性能分析命令
    perf_parser = subparsers.add_parser('performance', help='分析性能指标')
    perf_parser.add_argument('--hours', type=int, default=24, help='分析最近N小时的数据 (默认: 24)')
    
    # 导出报告命令
    report_parser = subparsers.add_parser('report', help='导出会话报告')
    report_parser.add_argument('session_id', help='会话ID')
    report_parser.add_argument('--output', help='输出文件路径')
    
    # 实时监控命令
    monitor_parser = subparsers.add_parser('monitor', help='实时监控日志')
    monitor_parser.add_argument('--interval', type=int, default=30, help='刷新间隔(秒) (默认: 30)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行对应命令
    if args.command == 'session':
        analyze_session(args.session_id)
    elif args.command == 'errors':
        find_errors(args.hours, args.limit)
    elif args.command == 'performance':
        analyze_performance(args.hours)
    elif args.command == 'report':
        export_session_report(args.session_id, args.output)
    elif args.command == 'monitor':
        monitor_realtime(args.interval)


if __name__ == "__main__":
    main()
