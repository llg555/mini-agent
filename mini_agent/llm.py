"""模型层：对 LLM API 的薄封装。

用 openai 官方 SDK，但通过 base_url 指向任意「OpenAI 兼容」服务——
DeepSeek、智谱 GLM、Qwen、Moonshot、本地 Ollama/vLLM 都实现了这套协议。
这就是为什么只需要学一套 SDK：OpenAI 的 chat completions + tools
已经是行业事实标准，换模型只改环境变量，不改代码。

配置从环境变量读取（支持 .env 文件），三个变量：
  LLM_API_KEY   必填
  LLM_BASE_URL  默认 https://api.deepseek.com
  LLM_MODEL     默认 deepseek-chat
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"


@dataclass
class AssistantMessage:
    """LLM 一次回复的归一化结构。

    把 openai SDK 返回的 pydantic 对象转成纯数据，好处有两个：
    1. agent.py 完全不依赖 SDK 的类型，换成任何模型服务都不用改循环逻辑；
    2. 测试时可以直接构造它来伪造 LLM 响应（见 tests/，不需要花一分钱）。

    tool_calls 用最简结构 [{"id", "name", "arguments"}]，其中 arguments 是
    JSON 字符串——和线上协议保持一致，方便理解真正的报文长什么样。
    """

    content: Optional[str] = None
    tool_calls: Optional[list] = None
    usage: Optional[dict] = None


def load_config() -> LLMConfig:
    """从环境变量加载配置；缺少 key 时给出可操作的报错。"""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise SystemExit(
            "\n未检测到 LLM_API_KEY。\n"
            "请把 .env.example 复制为 .env 并填入你的 key（项目根目录下）。\n"
            "获取方式（三选一）：\n"
            "  DeepSeek: https://platform.deepseek.com （充值 10 元够练很久）\n"
            "  智谱 GLM: https://open.bigmodel.cn    （glm-4-flash 模型免费）\n"
            "  OpenAI:   https://platform.openai.com\n"
        )
    return LLMConfig(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
    )


class LLMClient:
    """把「发消息、拿回复」封装成 chat() 一个方法，其余交给 agent 层。"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or load_config()
        # SDK 自带超时与限流重试（max_retries 默认 2），练习项目够用
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def chat(self, messages: list, tools: Optional[list] = None) -> AssistantMessage:
        """发起一次对话请求。messages 是完整对话历史，tools 是工具 schema 列表。"""
        kwargs = {"model": self.config.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in msg.tool_calls
            ]

        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }

        return AssistantMessage(content=msg.content, tool_calls=tool_calls, usage=usage)
