"""测试配置管理模块

提供安全的API密钥管理和测试配置
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TestConfig:
    """测试配置类"""
    
    # API配置
    api_key: str
    api_base_url: str = "https://api.siliconflow.cn/v1"
    api_timeout: int = 30
    api_max_retries: int = 3
    
    # 数据库配置
    test_db_path: str = "data/test_dialogues.db"
    use_in_memory_db: bool = False
    
    # 测试行为配置
    use_real_api: bool = True  # 是否使用真实API进行测试
    skip_slow_tests: bool = False  # 是否跳过慢速测试
    cleanup_after_tests: bool = True  # 测试后是否清理数据
    
    # 设备配置
    test_device_config: Dict[str, Any] = None
    
    # 并发测试配置
    max_concurrent_requests: int = 5
    load_test_duration: int = 10  # 秒
    
    def __post_init__(self):
        if self.test_device_config is None:
            self.test_device_config = {
                "rooms": ["客厅", "主卧", "次卧", "厨房", "书房"],
                "device_types": ["灯", "空调", "电视", "风扇"],
                "test_scenarios": [
                    "基本开关控制",
                    "属性调节",
                    "多设备协同",
                    "异常处理"
                ]
            }


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._config: Optional[TestConfig] = None
    
    def get_config(self) -> TestConfig:
        """获取测试配置"""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> TestConfig:
        """加载配置"""
        # 1. 优先从环境变量获取API密钥
        api_key = os.getenv("DIALOGUE_API_KEY")
        
        # 2. 如果环境变量不存在，尝试从本地配置文件读取
        if not api_key:
            api_key = self._load_api_key_from_file()
        
        # 3. 如果仍然没有，使用默认测试密钥（仅用于离线测试）
        if not api_key:
            api_key = "sk-test-key-for-offline-testing"
            self.logger.warning("未找到API密钥，使用默认测试密钥（功能受限）")
        
        # 从环境变量获取其他配置
        use_real_api = os.getenv("USE_REAL_API", "true").lower() == "true"
        skip_slow_tests = os.getenv("SKIP_SLOW_TESTS", "false").lower() == "true"
        use_in_memory_db = os.getenv("USE_IN_MEMORY_DB", "false").lower() == "true"
        
        return TestConfig(
            api_key=api_key,
            api_base_url=os.getenv("API_BASE_URL", "https://api.siliconflow.cn/v1"),
            api_timeout=int(os.getenv("API_TIMEOUT", "30")),
            api_max_retries=int(os.getenv("API_MAX_RETRIES", "3")),
            test_db_path=os.getenv("TEST_DB_PATH", "data/test_dialogues.db"),
            use_in_memory_db=use_in_memory_db,
            use_real_api=use_real_api,
            skip_slow_tests=skip_slow_tests,
            cleanup_after_tests=os.getenv("CLEANUP_AFTER_TESTS", "true").lower() == "true",
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5")),
            load_test_duration=int(os.getenv("LOAD_TEST_DURATION", "10"))
        )
    
    def _load_api_key_from_file(self) -> Optional[str]:
        """从本地文件加载API密钥"""
        key_files = [
            ".env",
            "tests/.env",
            "tests/config/.env",  # 添加测试配置目录
            "tests/.api_key",
            "config/api_key.txt",
            "tests/config/api_key.txt"
        ]
        
        for key_file in key_files:
            key_path = Path(key_file)
            if key_path.exists():
                try:
                    content = key_path.read_text(encoding='utf-8').strip()
                    # 支持.env格式
                    if "=" in content:
                        for line in content.split('\n'):
                            if line.startswith('DIALOGUE_API_KEY='):
                                return line.split('=', 1)[1].strip()
                    else:
                        # 纯文本格式
                        return content
                except Exception as e:
                    self.logger.warning(f"读取API密钥文件失败 {key_file}: {e}")
        
        return None
    
    def create_api_key_file(self, api_key: str):
        """创建API密钥文件"""
        config_dir = Path("tests/config")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建.env文件
        env_file = config_dir / ".env"
        env_file.write_text(f"DIALOGUE_API_KEY={api_key}\n")
        
        # 创建.gitignore确保不提交密钥
        gitignore_file = config_dir / ".gitignore"
        gitignore_content = """# 忽略API密钥文件
.env
api_key.txt
*.key
"""
        gitignore_file.write_text(gitignore_content)
        
        self.logger.info(f"API密钥已保存到 {env_file}")


# 全局配置管理器实例
config_manager = ConfigManager()


def get_test_config() -> TestConfig:
    """获取测试配置的便捷函数"""
    return config_manager.get_config()


def setup_api_key(api_key: str):
    """设置API密钥的便捷函数"""
    config_manager.create_api_key_file(api_key)


# 预设置用户提供的API密钥
if __name__ == "__main__":
    # 创建配置目录和API密钥文件
    # setup_api_key("sk-wkp")
    print(get_test_config())
