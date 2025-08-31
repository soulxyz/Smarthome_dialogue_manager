"""硅基流动API客户端模块

负责与硅基流动API进行通信，支持重试机制和错误处理。
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class APIResponse:
    """API响应数据结构"""
    success: bool
    content: str
    usage: Dict[str, int]
    model: str
    response_time: float
    error_message: str = ""
    raw_response: Dict = None


class SiliconFlowClient:
    """硅基流动API客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1"):
        """初始化API客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = "zai-org/GLM-4.5-Air"
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SmartHome-DialogueManager/1.0"
        })
        
        # 配置参数
        self.timeout = 30  # 请求超时时间（秒）
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 1  # 重试延迟（秒）
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized SiliconFlow client with model: {self.model_id}")
    
    def chat_completion(self, messages: List[Dict], **kwargs) -> APIResponse:
        """调用聊天完成API
        
        Args:
            messages: 消息列表，符合OpenAI格式
            **kwargs: 其他API参数
            
        Returns:
            APIResponse: API响应结果
        """
        start_time = time.time()
        
        # 验证messages参数
        if not messages or len(messages) == 0:
            return APIResponse(
                success=False,
                content="",
                error_message="Messages cannot be empty",
                response_time=0
            )
        
        if len(messages) > 10:
            # 如果消息太多，只保留最近的10条
            messages = messages[-10:]
            self.logger.warning(f"Messages truncated to 10 items")
        
        # 验证每个message的格式
        validated_messages = []
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                self.logger.warning(f"Invalid message format: {msg}")
                continue
            if msg["role"] not in ["system", "user", "assistant"]:
                self.logger.warning(f"Invalid role: {msg['role']}")
                continue
            validated_messages.append({
                "role": msg["role"],
                "content": str(msg["content"])
            })
        
        if not validated_messages:
            return APIResponse(
                success=False,
                content="",
                error_message="No valid messages found",
                response_time=0
            )
        
        # 构建请求参数
        payload = {
            "model": self.model_id,
            "messages": validated_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": kwargs.get("stream", False)
        }
        
        # 添加其他可选参数
        optional_params = ["presence_penalty", "frequency_penalty", "stop", "n"]
        for param in optional_params:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        self.logger.info(f"Sending chat completion request with {len(messages)} messages")
        
        # 执行请求（带重试机制）
        response = self._make_request_with_retry(
            method="POST",
            endpoint="/chat/completions",
            payload=payload
        )
        
        response_time = time.time() - start_time
        
        if response.get("success", False):
            # 解析成功响应
            data = response["data"]
            content = ""
            usage = {}
            
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content", "")
                elif "delta" in choice:  # 流式响应
                    content = choice["delta"].get("content", "")
            
            if "usage" in data:
                usage = data["usage"]
            
            return APIResponse(
                success=True,
                content=content,
                usage=usage,
                model=data.get("model", self.model_id),
                response_time=response_time,
                raw_response=data
            )
        else:
            # 处理错误响应
            error_msg = response.get("error", "Unknown API error")
            self.logger.error(f"API request failed: {error_msg}")
            
            return APIResponse(
                success=False,
                content="",
                usage={},
                model=self.model_id,
                response_time=response_time,
                error_message=error_msg,
                raw_response=response
            )
    
    def _make_request_with_retry(self, method: str, endpoint: str, payload: Dict) -> Dict:
        """带重试机制的请求方法
        
        Args:
            method: HTTP方法
            endpoint: API端点
            payload: 请求载荷
            
        Returns:
            Dict: 响应结果
        """
        url = f"{self.base_url}{endpoint}"
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"API request attempt {attempt + 1}/{self.max_retries + 1}")
                
                if method.upper() == "POST":
                    response = self.session.post(
                        url,
                        json=payload,
                        timeout=self.timeout
                    )
                elif method.upper() == "GET":
                    response = self.session.get(
                        url,
                        params=payload,
                        timeout=self.timeout
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # 检查HTTP状态码
                if response.status_code == 200:
                    try:
                        data = response.json()
                        return {"success": True, "data": data}
                    except json.JSONDecodeError as e:
                        return {
                            "success": False,
                            "error": f"Invalid JSON response: {e}"
                        }
                elif response.status_code == 429:  # 速率限制
                    error_msg = f"Rate limit exceeded (HTTP {response.status_code})"
                    self.logger.warning(error_msg)
                    last_error = error_msg
                    
                    # 对于速率限制，使用更长的延迟
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2 ** attempt)  # 指数退避
                        self.logger.info(f"Retrying after {delay} seconds...")
                        time.sleep(delay)
                        continue
                elif response.status_code >= 500:  # 服务器错误
                    error_msg = f"Server error (HTTP {response.status_code}): {response.text}"
                    self.logger.warning(error_msg)
                    last_error = error_msg
                    
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                        continue
                else:  # 客户端错误（4xx）
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", response.text)
                    except:
                        error_msg = response.text
                    
                    return {
                        "success": False,
                        "error": f"Client error (HTTP {response.status_code}): {error_msg}"
                    }
                
            except requests.exceptions.Timeout:
                error_msg = f"Request timeout after {self.timeout} seconds"
                self.logger.warning(error_msg)
                last_error = error_msg
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                    
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error: {e}"
                self.logger.warning(error_msg)
                last_error = error_msg
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                    
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                self.logger.error(error_msg)
                last_error = error_msg
                break
        
        # 所有重试都失败了
        return {
            "success": False,
            "error": f"All retry attempts failed. Last error: {last_error}"
        }
    
    def test_connection(self) -> bool:
        """测试API连接
        
        Returns:
            bool: 连接是否成功
        """
        test_messages = [
            {"role": "system", "content": "你是一个测试助手。"},
            {"role": "user", "content": "请回复'连接测试成功'"}
        ]
        
        try:
            response = self.chat_completion(test_messages, max_tokens=50)
            if response.success and "成功" in response.content:
                self.logger.info("API connection test successful")
                return True
            else:
                self.logger.warning(f"API connection test failed: {response.error_message}")
                return False
        except Exception as e:
            self.logger.error(f"API connection test error: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """获取模型信息
        
        Returns:
            Dict: 模型信息
        """
        # 硅基流动API可能不支持模型信息查询，返回默认信息
        return {
            "model_id": self.model_id,
            "provider": "SiliconFlow",
            "description": "GLM-4.5-Air model via SiliconFlow API",
            "max_tokens": 8192,
            "supports_streaming": True
        }
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: 估算的token数量
        """
        # 简单的token估算：中文字符按1.5个token计算，英文单词按1个token计算
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_words = len(text.replace('，', ' ').replace('。', ' ').split()) - chinese_chars
        
        estimated_tokens = int(chinese_chars * 1.5 + english_words)
        return max(estimated_tokens, len(text) // 4)  # 最少按4个字符1个token计算
    
    def build_smart_home_prompt(self, user_input: str, context: Dict, intent: str = None) -> List[Dict]:
        """构建智能家居专用的提示消息
        
        Args:
            user_input: 用户输入
            context: 对话上下文
            intent: 识别的意图
            
        Returns:
            List[Dict]: 消息列表
        """
        messages = []
        
        # 系统提示
        system_prompt = """
你是一个智能家居助手，负责理解用户的指令并提供相应的控制建议。

你的能力包括：
1. 理解用户对家电设备的控制需求
2. 提供设备操作的具体步骤
3. 根据用户偏好给出个性化建议
4. 在指令不明确时主动询问澄清

请用简洁、友好的语言回复用户。
        """.strip()
        
        messages.append({"role": "system", "content": system_prompt})
        
        # 添加上下文信息
        if context.get("preferences"):
            context_info = f"用户偏好：{json.dumps(context['preferences'], ensure_ascii=False)}"
            messages.append({"role": "system", "content": context_info})
        
        if context.get("device_config"):
            device_info = f"设备配置：{json.dumps(context['device_config'], ensure_ascii=False)}"
            messages.append({"role": "system", "content": device_info})
        
        # 添加意图信息（如果有）
        if intent:
            intent_info = f"识别的用户意图：{intent}"
            messages.append({"role": "system", "content": intent_info})
        
        # 添加用户输入
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def update_config(self, **kwargs):
        """更新客户端配置
        
        Args:
            **kwargs: 配置参数
        """
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
        if "max_retries" in kwargs:
            self.max_retries = kwargs["max_retries"]
        if "retry_delay" in kwargs:
            self.retry_delay = kwargs["retry_delay"]
        if "model_id" in kwargs:
            self.model_id = kwargs["model_id"]
        
        self.logger.info(f"Updated client config: {kwargs}")
    
    def get_usage_statistics(self) -> Dict:
        """获取使用统计（需要在实际使用中累积）
        
        Returns:
            Dict: 使用统计信息
        """
        # TODO: 实现使用统计收集
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "average_response_time": 0.0
        }