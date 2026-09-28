"""mini-agent：一个不依赖任何框架的教学版 ReAct Agent。

分层（从下往上读）：
  tools.py  工具层 —— agent 的「手」：能做什么（函数 + 给 LLM 看的 schema）
  llm.py    模型层 —— agent 的「大脑」接口：LLM API 封装（OpenAI 兼容协议）
  agent.py  核心层 —— agent 的「循环」：ReAct 循环与消息历史管理
"""

from .agent import DEFAULT_SYSTEM_PROMPT, Agent
from .llm import AssistantMessage, LLMClient, LLMConfig, load_config
from .tools import ALL_TOOLS, Tool, tools_schemas

__all__ = [
    "Agent",
    "LLMClient",
    "LLMConfig",
    "AssistantMessage",
    "Tool",
    "ALL_TOOLS",
    "tools_schemas",
    "load_config",
    "DEFAULT_SYSTEM_PROMPT",
]
