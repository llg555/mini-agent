"""不需要 API key 的测试：用 FakeLLM 替身验证 agent 的核心逻辑。

思路：LLM 是 agent 的一个依赖，把它 mock 成「按剧本演」的替身，
循环控制、工具执行、消息历史这些真正属于我们的代码就能离线全覆盖。
这是 agent 工程化的关键一课——能离线测试，才敢重构。

运行：.venv/bin/python -m pytest tests/ -v
"""

import json

import pytest

from mini_agent import tools as tool_mod
from mini_agent.agent import Agent
from mini_agent.llm import AssistantMessage


class FakeLLM:
    """按给定剧本依次返回响应，同时记录收到的工具表和消息。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.seen_tools = None
        self.last_messages = None

    def chat(self, messages, tools=None):
        self.last_messages = messages
        self.seen_tools = tools
        self.calls += 1
        if not self.script:
            raise AssertionError("剧本演完了，agent 还在继续调用 LLM（没有正确终止）")
        return self.script.pop(0)


def make_agent(script, **kwargs):
    return Agent(FakeLLM(script), verbose=False, **kwargs)


def tool_call(id, name, arguments):
    return {"id": id, "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}


# ---------------------------------------------------------------------------
# 工具层单测
# ---------------------------------------------------------------------------


def test_calculate_basic():
    assert tool_mod.calculate("2+3*4") == "14"
    assert tool_mod.calculate("(1+2)**10") == "59049"
    assert tool_mod.calculate("10/4") == "2.5"


def test_calculate_rejects_dangerous_expressions():
    for evil in ["__import__('os').system('ls')", "().__class__", "1 if True else 2"]:
        with pytest.raises(ValueError):
            tool_mod.calculate(evil)


def test_file_tools_are_sandboxed():
    with pytest.raises(PermissionError):
        tool_mod.read_file("/etc/passwd")
    with pytest.raises(PermissionError):
        tool_mod.read_file("../../.zshrc")
    with pytest.raises(PermissionError):
        tool_mod.list_files("/etc")


def test_schema_shape_is_openai_standard():
    schemas = tool_mod.tools_schemas()
    for s in schemas:
        assert s["type"] == "function"
        for key in ("name", "description", "parameters"):
            assert key in s["function"]


# ---------------------------------------------------------------------------
# Agent loop 集成测试（FakeLLM）
# ---------------------------------------------------------------------------


def test_agent_executes_tools_and_finishes():
    agent = make_agent(
        [
            AssistantMessage(
                tool_calls=[tool_call("c1", "calculate", {"expression": "123456789*987654321"})]
            ),
            AssistantMessage(content="123456789 乘以 987654321 等于 121932631112635269。"),
        ]
    )
    answer = agent.run("123456789 乘以 987654321 是多少？")

    assert "121932631112635269" in answer
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_msg = agent.messages[3]
    assert tool_msg["tool_call_id"] == "c1"
    assert tool_msg["content"] == "121932631112635269"
    # 传给 LLM 的工具表是 OpenAI 标准格式
    assert agent.client.seen_tools[0]["function"]["name"] == "calculate"


def test_tool_errors_become_observations_not_crashes():
    agent = make_agent(
        [
            AssistantMessage(tool_calls=[tool_call("c1", "calculate", {"expression": "2+*3"})]),
            AssistantMessage(content="表达式有语法错误，我换个写法重算。"),
        ]
    )
    agent.run("算 2+3")

    observation = agent.messages[3]["content"]
    assert "工具执行出错" in observation  # 错误成了喂回模型的观察，程序没有崩溃


def test_unknown_tool_and_recovery():
    agent = make_agent(
        [
            AssistantMessage(tool_calls=[tool_call("c1", "no_such_tool", {})]),
            AssistantMessage(tool_calls=[tool_call("c2", "calculate", {"expression": "2+3"})]),
            AssistantMessage(content="5"),
        ]
    )
    agent.run("测试")

    assert "没有找到工具" in agent.messages[3]["content"]
    assert agent.messages[5]["content"] == "5"


def test_bad_json_arguments_get_error_observation():
    agent = make_agent(
        [
            AssistantMessage(
                tool_calls=[{"id": "c1", "name": "calculate", "arguments": "{broken json"}]
            ),
            AssistantMessage(content="好的，我修正参数。"),
        ]
    )
    agent.run("x")

    assert "不是合法 JSON" in agent.messages[3]["content"]


def test_max_turns_guard():
    endless = [
        AssistantMessage(tool_calls=[tool_call(f"c{i}", "calculate", {"expression": "1+1"})])
        for i in range(10)
    ]
    agent = make_agent(endless, max_turns=3)
    answer = agent.run("会死循环的问题")

    assert "强制停止" in answer
    assert agent.client.calls == 3  # 轮数被兜底限制，不会无限烧钱


def test_history_persists_across_runs():
    agent = make_agent(
        [
            AssistantMessage(tool_calls=[tool_call("c1", "calculate", {"expression": "6*7"})]),
            AssistantMessage(content="42"),
            AssistantMessage(content="你刚问的是 6*7，答案是 42"),
        ]
    )
    agent.run("6*7 等于几")
    agent.run("我刚才问了什么")

    assert agent.messages[0]["role"] == "system"
    assert agent.messages[-1]["role"] == "assistant"
    assert agent.client.calls == 3
