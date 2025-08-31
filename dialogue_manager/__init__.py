"""智能家居多轮对话管理引擎

本模块提供多轮对话管理、意图识别、上下文记忆等核心功能。
"""

__version__ = "0.1.0"
__author__ = "SmartHome DialogueManager Team"

from .api_client import SiliconFlowClient
from .engine import DialogueEngine
from .intent import IntentRecognizer
from .memory import MemoryManager

__all__ = ["DialogueEngine", "IntentRecognizer", "MemoryManager", "SiliconFlowClient"]
