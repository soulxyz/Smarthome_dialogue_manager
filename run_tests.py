#!/usr/bin/env python3
"""测试运行脚本

提供便捷的测试执行命令，支持不同类型的测试和代码质量检查。

使用方法:
    python run_tests.py --help
    python run_tests.py --unit
    python run_tests.py --integration
    python run_tests.py --all
    python run_tests.py --lint
    python run_tests.py --format
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """运行命令并返回是否成功"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} - 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失败 (退出码: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"❌ {description} - 命令未找到: {cmd[0]}")
        return False


def run_unit_tests():
    """运行单元测试"""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "not integration", "-v"]
    return run_command(cmd, "单元测试")


def run_integration_tests():
    """运行集成测试"""
    cmd = ["python", "-m", "pytest", "tests/test_integration.py", "-v"]
    return run_command(cmd, "集成测试")


def run_all_tests():
    """运行所有测试"""
    cmd = ["python", "-m", "pytest", "tests/", "-v", "--cov=dialogue_manager", "--cov-report=html"]
    return run_command(cmd, "所有测试 (含覆盖率)")


def run_lint():
    """运行代码检查"""
    success = True

    # Flake8 检查
    cmd = ["python", "-m", "flake8", "dialogue_manager/", "tests/", "ui/"]
    success &= run_command(cmd, "Flake8 代码检查")

    # MyPy 类型检查
    cmd = ["python", "-m", "mypy", "dialogue_manager/", "--ignore-missing-imports"]
    success &= run_command(cmd, "MyPy 类型检查")

    return success


def run_format_check():
    """检查代码格式"""
    success = True

    # Black 格式检查
    cmd = ["python", "-m", "black", "--check", "--diff", "."]
    success &= run_command(cmd, "Black 格式检查")

    # isort 导入排序检查
    cmd = ["python", "-m", "isort", "--check-only", "--diff", "."]
    success &= run_command(cmd, "isort 导入排序检查")

    return success


def format_code():
    """格式化代码"""
    success = True

    # Black 格式化
    cmd = ["python", "-m", "black", "."]
    success &= run_command(cmd, "Black 代码格式化")

    # isort 导入排序
    cmd = ["python", "-m", "isort", "."]
    success &= run_command(cmd, "isort 导入排序")

    return success


def run_security_check():
    """运行安全检查"""
    success = True

    # Bandit 安全检查
    cmd = ["python", "-m", "bandit", "-r", "dialogue_manager/"]
    success &= run_command(cmd, "Bandit 安全检查")

    # Safety 依赖安全检查
    cmd = ["python", "-m", "safety", "check"]
    success &= run_command(cmd, "Safety 依赖安全检查")

    return success


def main():
    parser = argparse.ArgumentParser(description="测试运行脚本")
    parser.add_argument("--unit", action="store_true", help="运行单元测试")
    parser.add_argument("--integration", action="store_true", help="运行集成测试")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--lint", action="store_true", help="运行代码检查")
    parser.add_argument("--format-check", action="store_true", help="检查代码格式")
    parser.add_argument("--format", action="store_true", help="格式化代码")
    parser.add_argument("--security", action="store_true", help="运行安全检查")
    parser.add_argument("--ci", action="store_true", help="运行CI流水线 (所有检查)")

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    success = True

    if args.unit:
        success &= run_unit_tests()

    if args.integration:
        success &= run_integration_tests()

    if args.all:
        success &= run_all_tests()

    if args.lint:
        success &= run_lint()

    if args.format_check:
        success &= run_format_check()

    if args.format:
        success &= format_code()

    if args.security:
        success &= run_security_check()

    if args.ci:
        print("\n🚀 运行完整CI流水线...")
        success &= run_format_check()
        success &= run_lint()
        success &= run_unit_tests()
        success &= run_integration_tests()
        success &= run_security_check()

    print(f"\n{'='*60}")
    if success:
        print("🎉 所有检查通过!")
        sys.exit(0)
    else:
        print("💥 部分检查失败，请查看上述输出")
        sys.exit(1)


if __name__ == "__main__":
    main()
