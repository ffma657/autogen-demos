"""AutoGen 冒烟测试：通过智谱 GLM-5 验证智能体能否正常收发消息。"""

from __future__ import annotations

import asyncio
import sys

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from conversation import print_conversation
from helper import create_model_client

# 切换模型时只需改这里，如 "deepseek-chat"、"glm-4.7"
MODEL_PRESET = "glm-5"
SHOW_THOUGHT = False  # 单 Agent 冒烟测试默认不打印思考


async def main() -> None:
    model_client = create_model_client(MODEL_PRESET)
    try:
        agent = AssistantAgent(
            name="test_assistant",
            model_client=model_client,
            description="用于冒烟测试的简单智能体",
            system_message="你是一个测试助手，请用一两句话简短友好地回复用户。",
        )

        result = await agent.run(task="你好，请做个简单自我介绍。")

        print(f"=== AutoGen + {MODEL_PRESET} 冒烟测试 ===")
        print_conversation(list(result.messages), show_thought=SHOW_THOUGHT)

        last_message = result.messages[-1]
        if not isinstance(last_message, TextMessage) or not last_message.content.strip():
            raise RuntimeError("智能体未返回有效文本回复")

        print(f"\n测试通过：{MODEL_PRESET} 模型已正常响应。")
    finally:
        await model_client.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
