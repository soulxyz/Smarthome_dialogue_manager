# 智能家居对话系统测试 Makefile

.PHONY: help install test test-unit test-integration test-all test-api test-internal clean lint format coverage docs

# 默认目标
.DEFAULT_GOAL := help

# 颜色定义
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# 帮助信息
help: ## 显示帮助信息
	@echo "$(BLUE)智能家居对话系统测试命令$(NC)"
	@echo ""
	@echo "$(GREEN)安装和环境$(NC):"
	@echo "  install          安装项目依赖"
	@echo "  install-dev      安装开发依赖"
	@echo ""
	@echo "$(GREEN)测试命令$(NC):"
	@echo "  test             运行所有测试（推荐）"
	@echo "  test-unit        仅运行单元测试"
	@echo "  test-integration 仅运行集成测试"
	@echo "  test-api         仅运行需要API的测试"
	@echo "  test-internal    仅运行内部逻辑测试（不消耗API）"
	@echo "  test-slow        运行包括慢速测试在内的所有测试"
	@echo "  test-performance 运行性能测试"
	@echo ""
	@echo "$(GREEN)测试分析$(NC):"
	@echo "  coverage         生成测试覆盖率报告"
	@echo "  test-stats       显示API使用统计"
	@echo "  performance-report 生成性能报告"
	@echo ""
	@echo "$(GREEN)代码质量$(NC):"
	@echo "  lint             运行代码检查"
	@echo "  format           格式化代码"
	@echo "  clean            清理临时文件"
	@echo ""
	@echo "$(GREEN)其他$(NC):"
	@echo "  docs             生成文档"
	@echo "  setup-api-key    设置API密钥"

# 安装依赖
install: ## 安装项目依赖
	@echo "$(BLUE)安装项目依赖...$(NC)"
	pip install -r requirements.txt

install-dev: ## 安装开发依赖
	@echo "$(BLUE)安装开发依赖...$(NC)"
	pip install -r requirements-dev.txt
	pip install -e .

# API密钥设置
setup-api-key: ## 交互式设置API密钥
	@echo "$(BLUE)设置API密钥...$(NC)"
	@read -p "请输入API密钥: " api_key; \
	python -c "from tests.config import setup_api_key; setup_api_key('$$api_key')"
	@echo "$(GREEN)API密钥设置完成$(NC)"

# 基础测试命令
test: ## 运行所有测试（跳过慢速测试，节约API调用）
	@echo "$(BLUE)运行所有测试（跳过慢速测试）...$(NC)"
	pytest --skip-slow -v

test-all: ## 运行所有测试（包括慢速测试）
	@echo "$(YELLOW)运行所有测试（包括慢速测试，可能消耗较多API调用）...$(NC)"
	@read -p "确定要运行所有测试吗？这可能消耗大量API调用 (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		pytest -v; \
	else \
		echo "$(RED)测试被取消$(NC)"; \
	fi

test-unit: ## 仅运行单元测试
	@echo "$(BLUE)运行单元测试...$(NC)"
	pytest --unit-only -v

test-integration: ## 仅运行集成测试
	@echo "$(BLUE)运行集成测试...$(NC)"
	pytest --integration-only -v

test-api: ## 仅运行需要API的测试
	@echo "$(YELLOW)运行API测试（将消耗API调用）...$(NC)"
	@read -p "确定要运行API测试吗？(y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		pytest -m api_required -v; \
	else \
		echo "$(RED)API测试被取消$(NC)"; \
	fi

test-internal: ## 仅运行内部逻辑测试（不消耗API）
	@echo "$(BLUE)运行内部逻辑测试（不消耗API）...$(NC)"
	pytest --internal-logic-only -v

test-slow: ## 运行慢速测试
	@echo "$(YELLOW)运行慢速测试...$(NC)"
	pytest -m slow -v

test-performance: ## 运行性能测试
	@echo "$(BLUE)运行性能测试...$(NC)"
	pytest --performance-only --performance-report -v

# 测试分析
coverage: ## 生成测试覆盖率报告
	@echo "$(BLUE)生成测试覆盖率报告...$(NC)"
	pytest --cov=dialogue_manager --cov-report=html --cov-report=term --skip-slow
	@echo "$(GREEN)覆盖率报告已生成: htmlcov/index.html$(NC)"

test-stats: ## 显示API使用统计
	@echo "$(BLUE)运行测试并显示API统计...$(NC)"
	pytest --api-stats --skip-slow -v

performance-report: ## 生成性能报告
	@echo "$(BLUE)生成性能报告...$(NC)"
	pytest --performance-only --performance-report -v

# 特定场景测试
test-device-manager: ## 测试设备管理器
	@echo "$(BLUE)测试设备管理器...$(NC)"
	pytest tests/test_real_device_manager.py -v

test-dialogue-engine: ## 测试对话引擎
	@echo "$(BLUE)测试对话引擎...$(NC)"
	pytest tests/test_real_dialogue.py -v

test-focus-omission: ## 测试焦点和省略消解
	@echo "$(BLUE)测试焦点和省略消解...$(NC)"
	pytest tests/test_real_focus_and_omission.py -v

test-edge-cases: ## 测试边界条件
	@echo "$(BLUE)测试边界条件...$(NC)"
	pytest tests/test_real_edge_cases.py -v

test-integration-scenarios: ## 测试集成场景
	@echo "$(BLUE)测试集成场景...$(NC)"
	pytest tests/test_real_integration.py -v

# 代码质量
lint: ## 运行代码检查
	@echo "$(BLUE)运行代码检查...$(NC)"
	python -m flake8 dialogue_manager tests --max-line-length=120 --ignore=E203,W503
	python -m pylint dialogue_manager --disable=C0114,C0115,C0116,R0903,R0913

format: ## 格式化代码
	@echo "$(BLUE)格式化代码...$(NC)"
	python -m black dialogue_manager tests --line-length=120
	python -m isort dialogue_manager tests --profile black

# 清理
clean: ## 清理临时文件
	@echo "$(BLUE)清理临时文件...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf data/test_*.db*
	rm -rf temp_*.db*
	@echo "$(GREEN)清理完成$(NC)"

# 文档
docs: ## 生成文档
	@echo "$(BLUE)生成文档...$(NC)"
	python -m pdoc dialogue_manager --html --output-dir docs/
	@echo "$(GREEN)文档已生成: docs/dialogue_manager/$(NC)"

# 环境检查
check-env: ## 检查测试环境
	@echo "$(BLUE)检查测试环境...$(NC)"
	@python -c "from tests.config import get_test_config; config = get_test_config(); print(f'API密钥: {\"✓\" if config.api_key and config.api_key != \"sk-test-key-for-offline-testing\" else \"✗\"}'); print(f'使用真实API: {\"✓\" if config.use_real_api else \"✗\"}'); print(f'数据库: {config.test_db_path}')"

# 快速测试（开发时使用）
quick-test: ## 快速测试（仅内部逻辑，用于开发）
	@echo "$(BLUE)快速测试（开发模式）...$(NC)"
	pytest --internal-logic-only -x -v

# CI/CD相关
ci-test: ## CI环境测试
	@echo "$(BLUE)CI环境测试...$(NC)"
	pytest --skip-slow --internal-logic-only --cov=dialogue_manager --cov-report=xml -v

# 调试相关
debug-test: ## 调试模式运行测试
	@echo "$(BLUE)调试模式运行测试...$(NC)"
	pytest --pdb --pdbcls=IPython.terminal.debugger:Pdb -s -v

# 并行测试（如果安装了pytest-xdist）
test-parallel: ## 并行运行测试
	@echo "$(BLUE)并行运行测试...$(NC)"
	pytest -n auto --skip-slow -v

# 生成测试报告
test-report: ## 生成详细测试报告
	@echo "$(BLUE)生成详细测试报告...$(NC)"
	pytest --html=test_report.html --self-contained-html --skip-slow -v
	@echo "$(GREEN)测试报告已生成: test_report.html$(NC)"

# 内存和性能分析
profile-test: ## 性能分析测试
	@echo "$(BLUE)性能分析测试...$(NC)"
	python -m cProfile -o test_profile.prof -m pytest tests/test_real_device_manager.py::TestDeviceManagerMemoryAndResources -v
	@echo "$(GREEN)性能分析结果: test_profile.prof$(NC)"

# 安全测试
security-test: ## 安全测试
	@echo "$(BLUE)运行安全测试...$(NC)"
	pytest tests/test_real_edge_cases.py::TestInputValidationAndSanitization -v

# 压力测试
stress-test: ## 压力测试
	@echo "$(YELLOW)运行压力测试（仅内部逻辑）...$(NC)"
	pytest -m "internal_logic and slow" -v

# 验证所有功能
validate: ## 验证所有核心功能
	@echo "$(BLUE)验证所有核心功能...$(NC)"
	pytest tests/test_real_integration.py::TestCompleteSystemValidation::test_end_to_end_system_validation -v

# 显示测试统计
test-info: ## 显示测试信息
	@echo "$(BLUE)测试统计信息:$(NC)"
	@echo "总测试数量: $$(find tests -name 'test_*.py' -exec grep -l 'def test_' {} \; | wc -l) 个文件"
	@echo "测试函数数量: $$(find tests -name 'test_*.py' -exec grep 'def test_' {} \; | wc -l) 个"
	@echo "API相关测试: $$(grep -r '@pytest.mark.skipif.*api' tests/ | wc -l) 个"
	@echo "内部逻辑测试: $$(grep -r 'internal_logic' tests/ | wc -l) 个"
	@echo "集成测试: $$(grep -r '@pytest.mark.integration' tests/ | wc -l) 个"