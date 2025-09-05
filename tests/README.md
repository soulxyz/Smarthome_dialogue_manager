
# 智能家居对话系统测试指南

本项目采用了全新的真实测试策略，彻底移除了原有的为了测试而测试的假代码，建立了基于真实组件和场景的测试体系。

## 🎯 测试理念

### 测试分类策略

我们根据API消耗成本将测试分为两类：

1. **需要API_KEY的测试** - 进行必要的功能测试，避免过度消耗
2. **内部逻辑测试** - 可以进行压力测试和大量测试

### 摒弃的旧模式

❌ **FakeAPIClient** - 完全脱离真实API逻辑  
❌ **FakeMemoryManager** - 绕过真实数据存储  
❌ **硬编码测试数据** - 不能反映真实使用场景  
❌ **Mock一切** - 测试变成自我欺骗  

✅ **真实组件** - 使用实际的API客户端和数据库  
✅ **真实场景** - 测试用户实际会遇到的情况  
✅ **智能分类** - API测试节约使用，内部逻辑可以压力测试  
✅ **配置管理** - 安全的API密钥管理  

## 🔧 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
make install

# 安装开发依赖
make install-dev
```

### 2. 设置API密钥

```bash
# 交互式设置
make setup-api-key

# 或手动创建
echo "your-api-key-here" > tests/.api_key
```

### 3. 运行测试

```bash
# 推荐：运行所有测试（跳过慢速测试）
make test

# 只运行内部逻辑测试（不消耗API）
make test-internal

# 运行API测试（会消耗API调用）
make test-api

# 运行集成测试
make test-integration
```

## 📁 测试文件结构

### 新的测试文件

```
tests/
├── config.py                    # 测试配置管理
├── fixtures.py                  # 真实测试夹具
├── conftest.py                  # pytest配置
├── test_real_unit.py            # 真实单元测试
├── test_real_device_manager.py  # 设备管理器测试
├── test_real_dialogue.py        # 对话系统测试
├── test_real_focus_and_omission.py  # 焦点和省略消解测试
├── test_real_edge_cases.py      # 边界条件测试
└── test_real_integration.py     # 集成测试
```

### 配置文件

- `tests/.api_key` - API密钥文件（被gitignore）
- `tests/.gitignore` - 测试专用忽略文件
- `pytest.ini` - pytest配置
- `requirements-dev.txt` - 开发依赖

## 🎮 测试命令详解

### 基础测试命令

```bash
# 运行所有测试（推荐日常使用）
make test                    # 跳过慢速测试，节约API

# 完整测试（谨慎使用）
make test-all               # 包括所有慢速测试，大量API消耗

# 按类型运行
make test-unit              # 单元测试
make test-integration       # 集成测试
make test-internal          # 内部逻辑（不消耗API）
make test-api              # API测试（消耗API）
```

### 特定模块测试

```bash
make test-device-manager    # 设备管理器
make test-dialogue-engine   # 对话引擎
make test-focus-omission    # 焦点和省略消解
make test-edge-cases        # 边界条件
make test-integration-scenarios  # 集成场景
```

### 性能和分析

```bash
make test-performance       # 性能测试
make coverage              # 覆盖率报告
make test-stats            # API使用统计
make performance-report     # 性能报告
```

### 开发辅助

```bash
make quick-test            # 快速测试（开发模式）
make debug-test            # 调试模式
make validate              # 验证核心功能
make check-env             # 检查测试环境
```

## 🏷️ 测试标记系统

### 测试类型标记

- `@pytest.mark.unit` - 单元测试
- `@pytest.mark.integration` - 集成测试
- `@pytest.mark.api_required` - 需要真实API
- `@pytest.mark.internal_logic` - 内部逻辑（可压力测试）
- `@pytest.mark.slow` - 慢速测试

### 功能模块标记

- `@pytest.mark.device_manager` - 设备管理器
- `@pytest.mark.dialogue_system` - 对话系统
- `@pytest.mark.focus_management` - 焦点管理
- `@pytest.mark.edge_cases` - 边界条件

### 使用示例

```python
@pytest.mark.internal_logic
def test_device_operations_stress():
    """内部逻辑压力测试 - 可以大量运行"""
    pass

@pytest.mark.api_required
@pytest.mark.skipif(not pytest.config.getoption("--use-real-api"), 
                   reason="跳过真实API测试")
def test_real_dialogue_with_api():
    """真实API测试 - 节约使用"""
    pass
```

## 📊 测试类型详解

### 1. 内部逻辑测试（可压力测试）

这类测试不消耗API调用，可以进行大量压力测试：

- **设备状态管理** - 1000个设备的创建、操作、查询
- **焦点实体管理** - 500次焦点切换和衰减
- **并发操作** - 多线程设备控制
- **内存管理** - 大量数据的创建和清理
- **错误恢复** - 各种异常情况的处理

```python
def test_device_creation_stress():
    """创建1000个设备，验证性能"""
    for i in range(1000):
        device = LightDevice(f"灯{i}", f"房间{i//20}")
        # 验证性能指标
```

### 2. API集成测试（必要功能测试）

这类测试会消耗API调用，只进行必要的功能验证：

- **基础对话流程** - 每种意图测试一次
- **错误处理** - API失败时的降级机制
- **澄清功能** - 模糊输入的处理
- **焦点追踪** - 真实对话中的焦点管理

```python
@pytest.mark.api_required
def test_basic_dialogue_flow(api_tracker):
    """基础对话流程 - 节约API使用"""
    essential_tests = [
        ("你好", "greeting"),
        ("打开客厅的灯", "device_control"),
        ("现在温度怎么样", "query_status"),
    ]
    # 只测试核心功能
```

## 🔧 配置管理

### API密钥管理

```python
# tests/config.py
class TestConfig:
    api_key: str                 # API密钥
    use_real_api: bool = True    # 是否使用真实API
    api_timeout: int = 30        # API超时时间
    max_concurrent_requests: int = 5  # 最大并发请求
```

### 环境变量

```bash
# 跳过API测试
export USE_REAL_API=false

# 使用内存数据库
export USE_IN_MEMORY_DB=true

# 设置API超时
export API_TIMEOUT=60
```

## 📈 性能基准

### 内部逻辑性能目标

- **设备操作**: >50次/秒
- **意图识别**: >100次/秒  
- **焦点管理**: >200次/秒
- **状态查询**: >500次/秒

### API调用效率目标

- **设备控制场景**: <0.5次API调用/操作
- **完整对话场景**: <1次API调用/轮次
- **集成测试**: <20次API调用/场景

## 🚀 持续集成

### GitHub Actions工作流

1. **内部逻辑测试** - 在所有Python版本上运行
2. **单元测试** - 快速验证核心功能
3. **性能测试** - 确保性能不降级
4. **代码质量** - 格式检查和安全扫描
5. **API集成测试** - 仅在有API密钥时运行

### 本地开发流程

```bash
# 开发时快速验证
make quick-test

# 提交前完整检查
make test
make lint
make coverage

# 发布前完整验证
make validate
```

## 🔍 故障排除

### 常见问题

1. **API密钥未设置**
   ```bash
   make setup-api-key
   # 或手动创建 tests/.api_key
   ```

2. **数据库权限问题**
   ```bash
   # 清理旧数据库文件
   make clean
   # 使用内存数据库
   export USE_IN_MEMORY_DB=true
   ```

3. **测试超时**
   ```bash
   # 跳过慢速测试
   make test --skip-slow
   # 增加超时时间
   export API_TIMEOUT=60
   ```

4. **并发冲突**
   ```bash
   # 单线程运行
   pytest -v
   # 而不是 pytest -n auto
   ```

### 调试技巧

```bash
# 单个测试调试
pytest tests/test_real_device_manager.py::test_specific_function -v -s

# 使用pdb调试
make debug-test

# 查看详细日志
pytest --log-cli-level=DEBUG -v

# 性能分析
make profile-test
```

## 📚 最佳实践

### 编写新测试

1. **选择合适的测试类型**
   - 内部逻辑 → 可以压力测试
   - 需要API → 最小化调用

2. **使用真实组件**
   ```python
   # ✅ 好的做法
   def test_with_real_components(device_manager, api_client):
       result = device_manager.perform_action("打开", "灯", "客厅")
       assert result["success"]
   
   # ❌ 避免的做法
   def test_with_fake_components():
       fake_manager = FakeDeviceManager()
       # ...
   ```

3. **合理使用标记**
   ```python
   @pytest.mark.internal_logic  # 可以压力测试
   @pytest.mark.device_manager  # 功能分类
   def test_device_operations():
       pass
   ```

4. **清理资源**
   ```python
   def test_with_cleanup(temp_db_path):
       # 测试代码
       pass  # temp_db_path自动清理
   ```

### 性能考虑

- **内部测试**: 追求高覆盖率和压力测试
- **API测试**: 追求关键路径覆盖
- **集成测试**: 追求真实场景模拟
- **边界测试**: 追求异常情况处理

## 📖 相关文档

- [API客户端文档](../dialogue_manager/api_client.py)
- [设备管理器文档](../dialogue_manager/device_manager.py)
- [对话引擎文档](../dialogue_manager/engine.py)
- [项目README](../README.md)

---

通过这套真实测试体系，我们确保了代码质量的同时，也建立了可持续的测试实践，让测试真正为产品质量服务，而不是为了测试而测试。
