"""离线演示：不花一分钱，看 agent 完整工作一次。

运行：.venv/bin/python tests/demo_no_api.py

剧情：问一个需要两步工具调用的问题，中间故意安排一次失败——
重点观察「错误如何被作为观察喂回模型、模型如何自我纠正」。
LLM 的回复是预录的（FakeLLM），但思考/动作/观察的循环是真实执行的。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_agent.agent import Agent
from mini_agent.llm import AssistantMessage


def tool_call(id, name, arguments):
    import json

    return {"id": id, "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}


SCRIPT = [
    # 第 1 轮：先查时间
    AssistantMessage(
        content="我需要先查当前时间，再做计算。",
        tool_calls=[tool_call("c1", "get_current_time", {"timezone": "Asia/Shanghai"})],
        usage={"prompt_tokens": 612, "completion_tokens": 45},
    ),
    # 第 2 轮：计算（故意写错表达式）
    AssistantMessage(
        tool_calls=[tool_call("c2", "calculate", {"expression": "(3526+8817)×*47"})],
        usage={"prompt_tokens": 813, "completion_tokens": 38},
    ),
    # 第 3 轮：读到错误观察后，自我纠正
    AssistantMessage(
        content="上一个表达式有语法错误，我重新写一次。",
        tool_calls=[tool_call("c3", "calculate", {"expression": "(3526+8817)*47"})],
        usage={"prompt_tokens": 896, "completion_tokens": 40},
    ),
    # 第 4 轮：给出最终回答
    AssistantMessage(
        content="结果如下：当前时间见第一轮工具返回；(3526+8817)*47 = 580121。",
        usage={"prompt_tokens": 978, "completion_tokens": 52},
    ),
]


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools=None):
        return self.script.pop(0)


def main():
    print("(离线演示：LLM 回复是预录的，循环与工具执行是真实运行的)\n")
    agent = Agent(FakeLLM(SCRIPT), verbose=True)
    answer = agent.run("现在是几点？另外帮我算一下 (3526+8817) * 47")
    print(f"\n🤖 回答: {answer}")


if __name__ == "__main__":
    main()
