"""RoundRobinGroupChat + 用户介入：每轮 Agent 说完后等待用户输入。

结构（AutoGen 官方推荐的人机循环模式）：

    外层 RoundRobinGroupChat
    ├── 内层 RoundRobinGroupChat [alice, bob]  ← Agent 互聊一轮后结束
    └── UserProxyAgent                         ← 等待用户输入

用户输入 exit / 结束 / quit 时终止；其他内容作为下一轮对话继续。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken

from conversation import print_run_stream
from helper import create_model_client

ALICE_PRESET = "glm-5"
BOB_PRESET = "deepseek-v4-flash"
USER_NAME = "user"
OPENING = "你们好！聊聊今天心情怎么样？"
AGENT_COUNT = 2  # alice + bob
SHOW_THOUGHT = True  # False 时不打印思考过程


async def user_input_func(prompt: str, cancellation_token: Optional[CancellationToken] = None) -> str:
    print("\n--- 轮次结束，请输入 ---")
    print("  · 输入 exit / 结束 / quit  → 结束对话")
    print("  · 输入其他内容            → 继续下一轮\n")
    return await asyncio.to_thread(input, "你: ")


def create_team() -> tuple[RoundRobinGroupChat, list]:
    alice_client = create_model_client(ALICE_PRESET, thinking=False)
    bob_client = create_model_client(BOB_PRESET)

    alice = AssistantAgent(
        name="alice",
        model_client=alice_client,
        description="乐观、爱提问的聊天伙伴（智谱 GLM）",
        system_message="你是 Alice，性格活泼开朗，用一两句话回复。",
    )
    bob = AssistantAgent(
        name="bob",
        model_client=bob_client,
        description="沉稳、爱思考的聊天伙伴（DeepSeek）",
        system_message="你是 Bob，性格沉稳理性，用一两句话回复。",
    )

    # agent_team = RoundRobinGroupChat(
    #     participants=[alice, bob],
    #     termination_condition=MaxMessageTermination(max_messages=AGENT_COUNT),
    # )

    user = UserProxyAgent(
        name=USER_NAME,
        description="人类用户，每轮结束后决定继续或退出",
        input_func=user_input_func,
    )

    exit_termination = (
        TextMentionTermination("exit", sources=[USER_NAME])
        | TextMentionTermination("结束", sources=[USER_NAME])
        | TextMentionTermination("quit", sources=[USER_NAME])
    )
    team = RoundRobinGroupChat(
        participants=[alice, bob, user],
        termination_condition=exit_termination,
    )
    return team, [alice_client, bob_client]


async def main() -> None:
    team, clients = create_team()
    try:
        print("=== RoundRobinGroupChat + 用户介入 ===")
        print(f"Alice: {ALICE_PRESET}  Bob: {BOB_PRESET}")

        # 使用 run_stream 实时打印；team.run() 会等全部结束后才返回，导致中途看不到 Agent 输出
        result = await print_run_stream(
            team.run_stream(task=OPENING),
            show_thought=SHOW_THOUGHT,
        )
        print(f"对话结束，原因: {result.stop_reason}")
    finally:
        for client in clients:
            await client.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
