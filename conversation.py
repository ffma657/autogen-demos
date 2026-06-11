"""对话输出工具：统一打印 Agent 消息，支持控制是否显示思考过程。"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import TypeVar

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    TextMessage,
    ThoughtEvent,
    UserInputRequestedEvent,
)

T = TypeVar("T", bound=TaskResult)


def print_message(
    message: BaseAgentEvent | BaseChatMessage,
    *,
    show_thought: bool = True,
) -> None:
    """打印单条消息。"""
    if isinstance(message, ThoughtEvent):
        if show_thought:
            print(f"[{message.source}/思考] {message.content}\n")
    elif isinstance(message, TextMessage):
        print(f"[{message.source}] {message.content}\n")


def print_conversation(
    messages: Sequence[BaseAgentEvent | BaseChatMessage],
    *,
    show_thought: bool = True,
) -> None:
    """按顺序打印对话消息列表。"""
    for message in messages:
        print_message(message, show_thought=show_thought)


async def print_run_stream(
    stream: AsyncGenerator[BaseAgentEvent | BaseChatMessage | T, None],
    *,
    show_thought: bool = True,
) -> T:
    """消费 run_stream 并实时打印消息，结束后返回 TaskResult。"""
    result: T | None = None
    async for message in stream:
        if isinstance(message, TaskResult):
            result = message  # type: ignore[assignment]
        elif isinstance(message, UserInputRequestedEvent):
            continue
        else:
            print_message(message, show_thought=show_thought)
    if result is None:
        raise RuntimeError("run_stream 未返回 TaskResult")
    return result


def print_agent_reply(
    agent_name: str,
    thought: str,
    output: str,
    *,
    show_thought: bool = True,
) -> None:
    """打印单个 Agent 的思考与最终输出。"""
    if show_thought and thought:
        print(f"[{agent_name}/思考] {thought}\n")
    if output:
        print(f"[{agent_name}] {output}\n")
    else:
        print(f"[{agent_name}] (无回复)\n")
