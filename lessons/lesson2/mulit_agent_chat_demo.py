"""双智能体简单对话示例。



说明：

- 智谱 GLM 要求 messages 严格交替（system -> user -> assistant -> user ...）。

- RoundRobinGroupChat 会把其他 Agent 的消息也当成 user 角色，容易触发 1213 错误。

- 因此这里用「轮流调用两个 Agent」的方式，更兼容智谱 API。

- 思考内容（ThoughtEvent）仅用于展示；传给下一个 Agent 的只有最终输出（TextMessage）。

"""



from __future__ import annotations



import asyncio

import sys

from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from autogen_agentchat.agents import AssistantAgent

from autogen_agentchat.base import TaskResult

from autogen_agentchat.messages import TextMessage, ThoughtEvent

from conversation import print_agent_reply
from helper import create_model_client



MODEL_PRESET = "glm-5"

ROUNDS = 2  # 对话轮数（每轮 alice、bob 各说一次）
SHOW_THOUGHT = True  # False 时不打印思考过程





@dataclass

class AgentReply:

    thought: str  # 推理/思考过程，仅展示用

    output: str   # 最终输出，传给下一个 Agent；无回复则为空字符串





def build_prompt(history: list[tuple[str, str]], speaker: str) -> str:

    """把历史对话拼成一条 user 消息，满足智谱对 prompt 格式的要求。"""

    lines = [f"{name}: {content}" for name, content in history]

    lines.append(f"请以 {speaker} 的身份回复上面内容，用一两句话。")

    return "\n".join(lines)





def parse_agent_reply(result: TaskResult, agent_name: str) -> AgentReply:

    """分离思考内容与最终输出。最终输出只取自 TextMessage。"""

    thought = ""

    output = ""



    for message in result.messages:

        if message.source != agent_name:

            continue

        if isinstance(message, ThoughtEvent):

            thought = message.content.strip()

        elif isinstance(message, TextMessage):

            output = message.content.strip()



    return AgentReply(thought=thought, output=output)





async def main() -> None:

    # alice 不进行深度思考
    alice_client = create_model_client(MODEL_PRESET, thinking=False)
    # bob 进行深度思考，并且输出 token 数为 4096
    bob_client = create_model_client(MODEL_PRESET, max_tokens=4096)

    try:

        alice = AssistantAgent(

            name="alice",

            model_client=alice_client,

            description="乐观、爱提问的聊天伙伴",

            system_message="你是 Alice，性格活泼开朗，用一两句话回复。",

        )

        bob = AssistantAgent(

            name="bob",

            model_client=bob_client,

            description="沉稳、爱思考的聊天伙伴",

            system_message="你是 Bob，性格沉稳理性，用一两句话回复。",

        )



        opening = "你们好！聊聊今天心情怎么样？"

        history: list[tuple[str, str]] = [("user", opening)]



        print(f"=== 双智能体对话（模型: {MODEL_PRESET}）===")

        print(f"[user] {opening}\n")



        for _ in range(ROUNDS):

            alice_result = await alice.run(

                task=build_prompt(history, "alice"),

                output_task_messages=False,

            )

            alice_reply = parse_agent_reply(alice_result, "alice")

            print_agent_reply(
                "alice", alice_reply.thought, alice_reply.output, show_thought=SHOW_THOUGHT
            )

            history.append(("alice", alice_reply.output))



            bob_result = await bob.run(

                task=build_prompt(history, "bob"),

                output_task_messages=False,

            )

            bob_reply = parse_agent_reply(bob_result, "bob")

            print_agent_reply(
                "bob", bob_reply.thought, bob_reply.output, show_thought=SHOW_THOUGHT
            )

            history.append(("bob", bob_reply.output))



        print("对话结束。")

    finally:

        await alice_client.close()

        await bob_client.close()





if __name__ == "__main__":

    if hasattr(sys.stdout, "reconfigure"):

        sys.stdout.reconfigure(encoding="utf-8")

    asyncio.run(main())


