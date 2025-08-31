.PHONY: help install test test-unit test-integration lint format format-check security clean coverage docs

# 默认目标
help:
	@echo "可用命令:"
	@echo "  install        - 安装依赖"
	@echo "  test           - 运行所有测试"
	@echo "  test-unit      - 运行单元测试"
	@echo "  test-integration - 运行集成测试"
	@echo "  lint           - 代码检查"
	@echo "  format         - 格式化代码"
	@echo "  format-check   - 检查代码格式"
	@echo "  security       - 安全检查"
	@echo "  coverage       - 生成覆盖率报告"
	@echo "  clean          - 清理临时文件"
	@echo "  ci             - 运行CI流水线"
	@echo "  docs           - 生成文档"

# 安装依赖
install:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-mock flake8 black isort mypy bandit safety

# 测试相关
test:
	python -m pytest tests/ -v --cov=dialogue_manager --cov-report=html

test-unit:
	python -m pytest tests/ -m "not integration" -v

test-integration:
	python -m pytest tests/test_integration.py -v

# 代码质量
lint:
	python -m flake8 dialogue_manager/ tests/ ui/
	python -m mypy dialogue_manager/ --ignore-missing-imports

format:
	python -m black .
	python -m isort .

format-check:
	python -m black --check --diff .
	python -m isort --check-only --diff .

# 安全检查
security:
	python -m bandit -r dialogue_manager/
	python -m safety check

# 覆盖率报告
coverage:
	python -m pytest tests/ --cov=dialogue_manager --cov-report=html --cov-report=term-missing
	@echo "覆盖率报告已生成: htmlcov/index.html"

# 清理
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# CI流水线
ci: format-check lint test security
	@echo "🎉 CI流水线执行完成!"

# 文档生成 (可选)
docs:
	@echo "文档生成功能待实现"

# 开发服务器
dev:
	streamlit run ui/app.py

# 数据库初始化
db-init:
	python -c "from dialogue_manager.memory import MemoryManager; MemoryManager().close()"

# 安装开发依赖
install-dev: install
	pip install pre-commit
	pre-commit install