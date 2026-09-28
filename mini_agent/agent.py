"""Agent 核心：把 LLM、工具和循环粘在一起的那个循环。

剥掉所有框架包装，agent 的本质就是这个循环：

    1. 把全部对话历史（system + user + assistant + tool 消息）发给 LLM，
       并附上可用工具的清单（schema）
    2. 看返回的 assistant 消息里有没有 tool_calls：
       有   -> 逐个执行工具，把结果作为 role="tool" 的消息追加进历史，回到 1
       没有 -> 这条纯文本就是最终答案，结束

这就是 ReAct（Reason + Act）：模型先「想」（要不要用工具、用什么参数），
再「动」（发起 tool call），拿到「观察」（工具结果）后继续想，直到能回答。
LangChain / LangGraph / OpenAI Agents SDK 的核心都是这个循环的工程化包装。

两个工程上重要的设计：
- 工具报错不向上抛异常，而是把错误文本作为观察喂回给模型，让它自己换姿势重试
  （比如表达式写错了就换个写法）——这是 agent 自我纠错能力的来源；
- max_turns 兜底防止死循环：agent 陷入「反复调工具不给答案」是真实存在的故障。
"""

from __future__ import annotations

import inspect
import json
from typing import Optional

from .llm import AssistantMessage
from .tools import ALL_TOOLS

DEFAULT_SYSTEM_PROMPT = (
    "你是一个运行在命令行里的助手 agent，可以调用工具来完成任务。"
    "凡是需要精确计算、当前时间、查看项目文件的事情，必须使用工具，不要凭记忆猜测。"
    "工具调用失败时，阅读错误信息，修正参数后重试，而不是编造结果。"
)

_ANSI = {
    "reset": "\033[0m",
    "cyan": "\033[36m",  # 思考
    "yellow": "\033[33m",  # 动作
    "magenta": "\033[35m",  # 观察
    "green": "\033[32m",  # 统计
}

OBSERVATION_DISPLAY_LIMIT = 300


def _c(color: str, text: str) -> str:
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


def _one_line(text: str, limit: int) -> str:
    """观察结果压成一行显示（工具可能返回多行文本）。"""
    squeezed = " ".join(text.split())
    return squeezed if len(squeezed) <= limit else squeezed[:limit] + " …(截断)"


class Agent:
    def __init__(
        self,
        client,
        tools: Optional[list] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 8,
        verbose: bool = True,
    ):
        self.client = client  # 任何有 .chat(messages, tools) 的对象
        self.tools = {t.name: t for t in (tools if tools is not None else ALL_TOOLS)}
        self.messages: list = [{"role": "system", "content": system_prompt}]
        self.max_turns = max_turns
        self.verbose = verbose
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def reset(self) -> None:
        """清空对话历史（system prompt 保留）。"""
        self.messages = [self.messages[0]]

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        prompt_before = self.total_prompt_tokens
        completion_before = self.total_completion_tokens

        for turn in range(1, self.max_turns + 1):
            reply = self.client.chat(self.messages, tools=self._tool_schemas())
            self._accumulate_usage(reply.usage)
            self.messages.append(self._assistant_to_history(reply))

            # 没有工具调用 => 这就是最终回答
            if not reply.tool_calls:
                self._report_usage(prompt_before, completion_before)
                return reply.content or ""

            # 模型可能边想边说话（content 和 tool_calls 同时存在）
            if self.verbose and reply.content:
                print(_c("cyan", f"[思考·第{turn}轮] ") + reply.content)

            for tc in reply.tool_calls:
                if self.verbose:
                    print(_c("yellow", f"[动作] {tc['name']}") + f"  参数: {tc['arguments']}")
                observation = self._execute(tc)
                if self.verbose:
                    shown = _one_line(observation, OBSERVATION_DISPLAY_LIMIT)
                    print(_c("magenta", "[观察] ") + shown)
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": observation}
                )

        # 走到这里说明 max_turns 轮全部在调工具，仍没有给出最终回答
        warning = (
            f"（连续 {self.max_turns} 轮工具调用仍未给出答案，已强制停止。"
            f"可调大 max_turns，或优化 system prompt / 工具描述）"
        )
        self.messages.append({"role": "assistant", "content": warning})
        self._report_usage(prompt_before, completion_before)
        return warning

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _tool_schemas(self) -> list:
        return [t.to_openai_schema() for t in self.tools.values()]

    def _execute(self, tool_call: dict) -> str:
        """执行单个工具调用；任何失败都转成字符串观察，绝不向上抛异常。

        把错误喂回给模型而不是崩溃，是 agent 和普通程序的关键区别：
        模型读到错误信息后通常能自己修正参数重试。
        """
        name = tool_call["name"]
        raw_args = tool_call.get("arguments") or "{}"

        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return f"参数不是合法 JSON（{e}），请重新生成正确的 JSON 参数"

        tool = self.tools.get(name)
        if tool is None:
            available = "、".join(self.tools)
            return f"没有找到工具 {name}，可用工具: {available}"

        try:
            return tool.execute(**args)
        except TypeError as e:
            sig = inspect.signature(tool.func)
            return f"参数与工具签名不匹配（{e}），正确签名: {name}{sig}"
        except Exception as e:
            return f"工具执行出错 [{type(e).__name__}]: {e}"

    @staticmethod
    def _assistant_to_history(reply: AssistantMessage) -> dict:
        """把 LLM 回复转成 OpenAI 协议格式的纯 dict 追加进历史。

        历史里全部是纯 dict，意味着整个 agent 不依赖任何 SDK 类型，
        打印、序列化、测试都直接可用。
        """
        msg: dict = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in reply.tool_calls
            ]
        return msg

    def _accumulate_usage(self, usage: Optional[dict]) -> None:
        if usage:
            self.total_prompt_tokens += usage.get("prompt_tokens", 0)
            self.total_completion_tokens += usage.get("completion_tokens", 0)

    def _report_usage(self, prompt_before: int, completion_before: int) -> None:
        if not self.verbose:
            return
        p = self.total_prompt_tokens - prompt_before
        c = self.total_completion_tokens - completion_before
        print(
            _c("green", f"[统计] 本次消耗 tokens: 输入 {p} / 输出 {c}")
            + _c(
                "green",
                f"；累计: 输入 {self.total_prompt_tokens} / 输出 {self.total_completion_tokens}",
            )
        )
