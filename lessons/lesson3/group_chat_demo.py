"""RoundRobinGroupChat 双智能体对话示例（Alice=GLM，Bob=DeepSeek）。

与 mulit_agent_chat_demo.py（手动轮流调用）对比：
- 本文件：使用 AutoGen 内置 RoundRobinGroupChat 编排发言顺序。
- 手动版：自行调用 agent.run()，把历史拼成一条 user 消息，更兼容纯 GLM 多 Agent。

混合模型说明：
- Alice 用智谱 GLM，Bob 用 DeepSeek。
- Bob 发言时会收到「user + alice」两条 user 消息，DeepSeek 通常可接受。
- Alice 再次发言时通常只收到 Bob 的最新一条，GLM 不易触发 1213。
- 若两个 Agent 都用 GLM，GroupChat 更容易报错，建议改用手动版 demo。

"""

from __future__ import annotations

import asyncio
import sys

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from conversation import print_conversation
from helper import create_model_client

ALICE_PRESET = "glm-5"
BOB_PRESET = "deepseek-v4-flash"
MAX_TURNS = 4  # Agent 发言次数；2=各说1次，4=各说2次
OPENING = "你们好！聊聊今天心情怎么样？"
SHOW_THOUGHT = True  # False 时不打印思考过程


def create_agents() -> tuple[AssistantAgent, AssistantAgent, list]:
    """创建 Agent 并返回 (alice, bob, clients) 便于统一关闭连接。"""
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
    return alice, bob, [alice_client, bob_client]


async def main() -> None:
    alice, bob, clients = create_agents()
    try:
        team = RoundRobinGroupChat(
            participants=[alice, bob],
            termination_condition=MaxMessageTermination(max_messages=6),
            max_turns=MAX_TURNS,
        )

        print("=== RoundRobinGroupChat 双智能体对话 ===")
        print(f"Alice: {ALICE_PRESET} (thinking=False)")
        print(f"Bob:   {BOB_PRESET}")
        print()

        result = await team.run(task=OPENING)

        print_conversation(list(result.messages), show_thought=SHOW_THOUGHT)
        print(f"对话结束，原因: {result.stop_reason}")
    finally:
        for client in clients:
            await client.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
