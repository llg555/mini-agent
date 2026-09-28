# mini-agent：手写一个不依赖任何框架的 ReAct Agent

一个教学项目：用约 400 行可读的 Python，从零实现一个能调用工具、能自我纠错的
命令行 Agent。不使用 LangChain / LlamaIndex 等任何框架——写完你就理解了
所有 agent 框架背后的本质，面试聊 agent 原理时可以直接拿它当例子。

## 你会学到什么（也是常见面试考点）

1. **Agent loop 的本质**：一个 while 循环 + 一个消息列表，没有魔法
2. **Function calling 协议**：工具的 JSON Schema、tool_calls、role="tool" 消息的真实格式
3. **错误即观察**：工具失败不崩溃，喂回给模型让它自我纠正（self-correction）
4. **安全边界**：AST 白名单求值（替代 eval）、文件沙箱、指数大小限制
5. **离线测试 agent**：用 FakeLLM 替身验证循环逻辑，不花一分钱 API 费
6. **成本意识**：每轮对话的 token 消耗统计

## 架构

```
你 ──▶ Agent Loop (agent.py)
          │
          │  ① 带上全部历史 + 工具清单，调用 LLM (llm.py)
          ▼
        LLM
          │ ② 返回 assistant 消息
          │
          ├── 有 tool_calls ──▶ ③ 执行工具 (tools.py)
          │                        │
          │                        ▼
          │        ④ 结果作为 role="tool" 消息入历史 ──▶ 回到 ①
          │
          └── 纯文本 ──▶ 这就是最终答案，结束
```

一次完整对话后，消息历史长这样（这就是线上协议的真实格式）：

```json
[
  {"role": "system",    "content": "你是一个运行在命令行里的助手 agent…"},
  {"role": "user",      "content": "123456789 乘 987654321 是多少"},
  {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
      "function": {"name": "calculate", "arguments": "{\"expression\": \"123456789*987654321\"}"}}]},
  {"role": "tool",      "tool_call_id": "c1", "content": "121932631112635269"},
  {"role": "assistant", "content": "等于 121932631112635269"}
]
```

注意两件事：**整个历史每次都完整发给模型**（这就是为什么长对话越来越贵，
也是"上下文工程"要解决的问题）；**工具结果是纯字符串**（模型只消费文本）。

## 快速开始

```bash
cd mini-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) 先看离线演示（不需要 API key，看懂 ReAct 循环）
python tests/demo_no_api.py

# 2) 跑测试（同样不需要 key，10 个用例覆盖循环逻辑）
pytest tests/ -v

# 3) 接真模型
cp .env.example .env   # 编辑 .env，填入 LLM_API_KEY（DeepSeek/智谱/OpenAI 三选一）
python main.py
```

接真模型后建议依次问它（每个都能触发不同的工具使用模式）：

- `算一下 123456789 乘以 987654321 等于多少`（单工具）
- `现在几点了？纽约现在又是几点？`（时区参数）
- `看看这个项目目录里有什么文件，然后读一下 README.md`（多步链式调用）
- 故意刁难：`帮我把系统文件 /etc/hosts 读出来`（观察沙箱如何拦截并让模型改口）

## 目录结构

```
mini-agent/
├── main.py                      # 交互入口
├── mini_agent/
│   ├── tools.py                 # 工具层：函数 + 自动生成 schema + 沙箱
│   ├── llm.py                   # 模型层：OpenAI 兼容 API 封装
│   └── agent.py                 # 核心：ReAct 循环、错误回喂、token 统计
├── tests/
│   ├── test_agent_no_api.py     # 离线单测（FakeLLM）
│   └── demo_no_api.py           # 离线演示脚本
├── .env.example                 # 配置模板（DeepSeek / 智谱 / OpenAI）
└── conftest.py
```

## 设计要点（面试谈资）

| 设计 | 为什么 |
|---|---|
| docstring 即工具描述 | 函数是唯一真相来源，schema 从签名自动推导，不会对不上 |
| 错误喂回而不抛出 | agent 的鲁棒性来自"模型读到错误能自己改"，而不是程序崩溃 |
| AST 白名单求值 | `eval()` 模型给的字符串 = 开放任意代码执行 |
| 路径沙箱 + resolve | 防止 `../` 和符号链接逃逸出项目目录 |
| 消息历史纯 dict | 不依赖 SDK 类型，可序列化、可测试、可迁移 |
| FakeLLM 测试 | LLM 是依赖不是逻辑，mock 掉才能离线验证"我们的代码" |
| max_turns 兜底 | "反复调工具不给答案"是真实的 agent 故障模式，必须设防 |

## 练习（自己动手改，才算真的学会）

1. **⭐ 加一个 `write_file` 工具**：注意沿用沙箱校验，写入后返回字节数
2. **⭐ 给 main.py 加 `/clear` 命令**：调用 `agent.reset()` 清空历史
3. **⭐⭐ 上下文管理**：verbose 里打印每轮的历史消息数，感受上下文增长；
   实现历史超过 N 条时把最旧的工具结果替换成 "(已截断)" —— 这就是上下文工程雏形
4. **⭐⭐ 换个"人设"**：改 system_prompt 让它变成一个只帮你管理文件的 agent，
   观察同样的工具、不同的 prompt，行为如何变化
5. **⭐⭐⭐ 异步化**：用 asyncio 改造（LLM 调用和工具执行天然是 IO 密集）
6. **⭐⭐⭐ 迈向 MCP**：把工具层抽成独立进程的 MCP server，本 agent 改为
   通过 MCP 协议调工具——这是下一个项目的自然起点

## 下一步项目方向

- **deep research agent**：加搜索工具 + 网页阅读工具，让它写调研报告（练 RAG）
- **coding agent**：加执行 Python 代码的工具（注意沙箱！），让它自己写代码、跑测试、修 bug
- **MCP 化**：把本项目工具层变成标准的 MCP server，接给 Claude Code / ZCode 这类客户端用
