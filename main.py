"""mini-agent 交互入口。

运行：python main.py
配置：复制 .env.example 为 .env 并填入 LLM_API_KEY（详见 README.md）
"""

from dotenv import load_dotenv

from mini_agent import Agent, LLMClient

SUGGESTIONS = [
    "算一下 123456789 乘以 987654321 等于多少",
    "现在几点了？纽约现在又是几点？",
    "看看这个项目目录里有什么文件，然后读一下 README.md 的前几行",
    "一个矩形长 3526 米、宽 47 米，另一个长 8817 米、宽 3 米，哪个大？大多少？",
]


def main() -> None:
    # 把 .env 里的配置加载进环境变量；只要在创建 LLMClient 之前执行即可
    load_dotenv()

    print("=" * 56)
    print(" mini-agent —— 手写的 ReAct Agent（无框架）")
    print("=" * 56)

    try:
        client = LLMClient()
    except SystemExit as e:
        print(e)
        return

    agent = Agent(client)
    print("试试这些问题：")
    for s in SUGGESTIONS:
        print(f"  · {s}")
    print("输入 exit 或按 Ctrl-C 退出\n")

    while True:
        try:
            question = input("你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break

        print()
        try:
            answer = agent.run(question)
        except Exception as e:
            # 网络波动导致的调用失败最常见：提示后继续循环，对话历史不丢
            print(f"\n⚠️ 这次调用失败了（多半是网络波动）：{type(e).__name__}: {e}")
            print("   直接重新问一遍就行。\n")
            continue
        print(f"\n🤖 回答: {answer}\n")


if __name__ == "__main__":
    main()
