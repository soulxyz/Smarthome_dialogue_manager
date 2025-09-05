"""pytest配置文件

全局测试配置、夹具和钩子函数
"""

import pytest
import os
import sys
import logging
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.config import get_test_config


def pytest_addoption(parser):
    """添加命令行选项"""
    # API测试控制
    parser.addoption(
        "--use-real-api", 
        action="store_true", 
        default=True,
        help="使用真实API进行测试（默认开启）"
    )
    parser.addoption(
        "--skip-api-tests", 
        action="store_true", 
        default=False,
        help="跳过所有需要API的测试"
    )
    
    # 性能测试控制
    parser.addoption(
        "--skip-slow", 
        action="store_true", 
        default=False,
        help="跳过慢速测试"
    )
    parser.addoption(
        "--performance-only", 
        action="store_true", 
        default=False,
        help="只运行性能测试"
    )
    
    # 测试类型控制
    parser.addoption(
        "--unit-only", 
        action="store_true", 
        default=False,
        help="只运行单元测试"
    )
    parser.addoption(
        "--integration-only", 
        action="store_true", 
        default=False,
        help="只运行集成测试"
    )
    parser.addoption(
        "--internal-logic-only", 
        action="store_true", 
        default=False,
        help="只运行内部逻辑测试（不消耗API）"
    )
    
    # 输出控制
    parser.addoption(
        "--api-stats", 
        action="store_true", 
        default=False,
        help="显示API使用统计"
    )
    parser.addoption(
        "--performance-report", 
        action="store_true", 
        default=False,
        help="生成性能报告"
    )


def pytest_configure(config):
    """配置pytest"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置测试相关的日志级别
    logging.getLogger("dialogue_manager").setLevel(logging.INFO)
    logging.getLogger("tests").setLevel(logging.INFO)
    
    # 如果跳过API测试，设置环境变量
    if config.getoption("--skip-api-tests"):
        os.environ["USE_REAL_API"] = "false"
        os.environ["SKIP_API_TESTS"] = "true"
    
    # 性能测试配置
    if config.getoption("--performance-only"):
        os.environ["PERFORMANCE_ONLY"] = "true"


def pytest_collection_modifyitems(config, items):
    """修改测试收集结果"""
    # 根据命令行选项修改测试
    
    if config.getoption("--skip-api-tests"):
        skip_api = pytest.mark.skip(reason="跳过API测试")
        for item in items:
            if "api_required" in item.keywords:
                item.add_marker(skip_api)
    
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="跳过慢速测试")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if config.getoption("--unit-only"):
        skip_non_unit = pytest.mark.skip(reason="只运行单元测试")
        for item in items:
            if "unit" not in item.keywords:
                item.add_marker(skip_non_unit)
    
    if config.getoption("--integration-only"):
        skip_non_integration = pytest.mark.skip(reason="只运行集成测试")
        for item in items:
            if "integration" not in item.keywords:
                item.add_marker(skip_non_integration)
    
    if config.getoption("--internal-logic-only"):
        skip_api_dependent = pytest.mark.skip(reason="只运行内部逻辑测试")
        for item in items:
            if "api_required" in item.keywords:
                item.add_marker(skip_api_dependent)
    
    if config.getoption("--performance-only"):
        skip_non_performance = pytest.mark.skip(reason="只运行性能测试")
        for item in items:
            if "performance" not in item.keywords and "slow" not in item.keywords:
                item.add_marker(skip_non_performance)


@pytest.fixture(scope="session")
def test_config():
    """全局测试配置夹具"""
    config = get_test_config()
    
    # 根据命令行选项调整配置
    if hasattr(pytest, "current_request"):
        request = pytest.current_request()
        if request.config.getoption("--skip-api-tests"):
            config.use_real_api = False
    
    return config


@pytest.fixture(scope="session")
def api_stats_collector():
    """API统计收集器"""
    class APIStatsCollector:
        def __init__(self):
            self.total_calls = 0
            self.successful_calls = 0
            self.failed_calls = 0
            self.total_tokens = 0
            self.total_time = 0.0
            self.calls_by_test = {}
        
        def record_call(self, test_name, success, tokens=0, duration=0.0):
            """记录API调用"""
            self.total_calls += 1
            if success:
                self.successful_calls += 1
            else:
                self.failed_calls += 1
            
            self.total_tokens += tokens
            self.total_time += duration
            
            if test_name not in self.calls_by_test:
                self.calls_by_test[test_name] = {
                    "calls": 0, "tokens": 0, "time": 0.0
                }
            
            self.calls_by_test[test_name]["calls"] += 1
            self.calls_by_test[test_name]["tokens"] += tokens
            self.calls_by_test[test_name]["time"] += duration
        
        def get_summary(self):
            """获取统计摘要"""
            return {
                "total_calls": self.total_calls,
                "success_rate": self.successful_calls / max(1, self.total_calls),
                "total_tokens": self.total_tokens,
                "total_time": self.total_time,
                "avg_time_per_call": self.total_time / max(1, self.total_calls),
                "calls_by_test": self.calls_by_test
            }
    
    return APIStatsCollector()


@pytest.fixture(scope="session")
def performance_collector():
    """性能数据收集器"""
    class PerformanceCollector:
        def __init__(self):
            self.test_performances = {}
        
        def record_performance(self, test_name, metrics):
            """记录性能数据"""
            self.test_performances[test_name] = metrics
        
        def get_summary(self):
            """获取性能摘要"""
            if not self.test_performances:
                return {}
            
            all_durations = []
            all_throughputs = []
            
            for metrics in self.test_performances.values():
                if "duration" in metrics:
                    all_durations.append(metrics["duration"])
                if "throughput" in metrics:
                    all_throughputs.append(metrics["throughput"])
            
            summary = {
                "total_tests": len(self.test_performances),
                "test_details": self.test_performances
            }
            
            if all_durations:
                summary.update({
                    "avg_duration": sum(all_durations) / len(all_durations),
                    "max_duration": max(all_durations),
                    "min_duration": min(all_durations)
                })
            
            if all_throughputs:
                summary.update({
                    "avg_throughput": sum(all_throughputs) / len(all_throughputs),
                    "max_throughput": max(all_throughputs),
                    "min_throughput": min(all_throughputs)
                })
            
            return summary
    
    return PerformanceCollector()


@pytest.fixture(autouse=True)
def test_isolation():
    """测试隔离夹具 - 每个测试前后执行"""
    # 测试前准备
    start_time = time.time()
    
    yield
    
    # 测试后清理
    duration = time.time() - start_time
    
    # 如果测试耗时过长，记录警告
    if duration > 30:
        logging.warning(f"Test took {duration:.2f} seconds - consider optimization")


def pytest_sessionstart(session):
    """测试会话开始时执行"""
    print("\n" + "="*80)
    print("开始执行真实智能家居对话系统测试")
    print("="*80)
    
    # 检查配置
    config = get_test_config()
    print(f"API密钥配置: {'✓' if config.api_key and config.api_key != 'sk-test-key-for-offline-testing' else '✗'}")
    print(f"使用真实API: {'✓' if config.use_real_api else '✗'}")
    print(f"数据库配置: {'内存数据库' if config.use_in_memory_db else config.test_db_path}")
    
    # 显示测试配置
    test_options = []
    if session.config.getoption("--skip-api-tests"):
        test_options.append("跳过API测试")
    if session.config.getoption("--skip-slow"):
        test_options.append("跳过慢速测试")
    if session.config.getoption("--unit-only"):
        test_options.append("仅单元测试")
    if session.config.getoption("--integration-only"):
        test_options.append("仅集成测试")
    if session.config.getoption("--internal-logic-only"):
        test_options.append("仅内部逻辑测试")
    
    if test_options:
        print(f"测试选项: {', '.join(test_options)}")
    
    print("="*80 + "\n")


def pytest_sessionfinish(session, exitstatus):
    """测试会话结束时执行"""
    print("\n" + "="*80)
    print("测试会话完成")
    
    # 显示API统计（如果请求）
    if session.config.getoption("--api-stats"):
        if hasattr(session, 'api_stats_collector'):
            stats = session.api_stats_collector.get_summary()
            print(f"\nAPI使用统计:")
            print(f"  总调用次数: {stats['total_calls']}")
            print(f"  成功率: {stats['success_rate']:.1%}")
            print(f"  总Token数: {stats['total_tokens']}")
            print(f"  总耗时: {stats['total_time']:.2f}秒")
            print(f"  平均耗时: {stats['avg_time_per_call']:.2f}秒/次")
    
    # 显示性能报告（如果请求）
    if session.config.getoption("--performance-report"):
        if hasattr(session, 'performance_collector'):
            perf = session.performance_collector.get_summary()
            if perf:
                print(f"\n性能测试报告:")
                print(f"  测试数量: {perf['total_tests']}")
                if "avg_duration" in perf:
                    print(f"  平均耗时: {perf['avg_duration']:.2f}秒")
                if "avg_throughput" in perf:
                    print(f"  平均吞吐量: {perf['avg_throughput']:.2f}/秒")
    
    # 显示退出状态
    status_msg = {
        0: "所有测试通过 ✓",
        1: "部分测试失败 ✗", 
        2: "测试被中断 ⚠",
        3: "内部错误 ❌",
        4: "pytest使用错误 ❌",
        5: "未发现测试 ⚠"
    }.get(exitstatus, f"未知状态 ({exitstatus})")
    
    print(f"退出状态: {status_msg}")
    print("="*80)


def pytest_runtest_setup(item):
    """每个测试执行前的设置"""
    # 检查测试标记和环境要求
    if "requires_network" in item.keywords:
        # 检查网络连接
        try:
            import requests
            requests.get("https://httpbin.org/get", timeout=5)
        except Exception:
            pytest.skip("需要网络连接")
    
    if "memory_intensive" in item.keywords:
        # 检查可用内存
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < 2:  # 小于2GB可用内存
                pytest.skip("内存不足，需要至少2GB可用内存")
        except ImportError:
            pass  # 如果没有psutil，跳过检查


def pytest_runtest_teardown(item, nextitem):
    """每个测试执行后的清理"""
    # 强制垃圾回收
    import gc
    gc.collect()


@pytest.fixture
def test_name(request):
    """获取当前测试名称的夹具"""
    return request.node.name


# 全局异常处理
def pytest_exception_interact(node, call, report):
    """处理测试异常"""
    if report.failed:
        logging.error(f"测试失败: {node.name}")
        logging.error(f"异常信息: {report.longrepr}")


# 为了向后兼容，导入所有原有夹具
from tests.fixtures import *
