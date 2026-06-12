"""猜拳游戏：两个 Agent 按规则对战，三局两胜。

设计要点：
- 游戏规则由代码编排器强制执行（非 GroupChat 自由对话）。
- 每局双方同时出拳（并行调用，互不可见对方本局选择）。
- Agent 只需回复「石头 / 剪刀 / 布」之一；解析失败会要求重试。
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, ThoughtEvent

from conversation import print_agent_reply
from helper import create_model_client

Move = Literal["石头", "剪刀", "布"]
Winner = Literal["alice", "bob", "draw"]

ALICE_PRESET = "glm-5"
BOB_PRESET = "deepseek-v4-flash"
WINS_TO_WIN = 2
MAX_RETRIES = 2
SHOW_THOUGHT = True

MOVE_EMOJI = {"石头": "✊", "剪刀": "✌️", "布": "✋"}
BEATS: dict[Move, Move] = {"石头": "剪刀", "剪刀": "布", "布": "石头"}

GAME_RULES = """\
猜拳规则：
1. 可选：石头、剪刀、布
2. 石头 胜 剪刀，剪刀 胜 布，布 胜 石头
3. 三局两胜，先赢 2 局者获胜
4. 每轮只回复一个词：石头、剪刀 或 布（不要解释、不要标点）"""


@dataclass
class AgentReply:
    thought: str
    output: str


@dataclass
class GameState:
    alice_wins: int = 0
    bob_wins: int = 0
    round_num: int = 0
    last_summary: str = "（第一局，尚无历史）"


def parse_agent_reply(result: TaskResult, agent_name: str) -> AgentReply:
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


def parse_move(text: str) -> Move | None:
    """从 Agent 回复中解析出拳。"""
    cleaned = text.strip().replace("，", "").replace("。", "").replace("！", "")
    if cleaned in ("石头", "剪刀", "布"):
        return cleaned

    lower = cleaned.lower()
    english_map = {"rock": "石头", "scissors": "剪刀", "paper": "布"}
    for en, cn in english_map.items():
        if en in lower:
            return cn  # type: ignore[return-value]

    for move in ("剪刀", "石头", "布"):
        if move in cleaned:
            return move  # type: ignore[return-value]
    return None


def judge(alice_move: Move, bob_move: Move) -> Winner:
    if alice_move == bob_move:
        return "draw"
    if BEATS[alice_move] == bob_move:
        return "alice"
    return "bob"


def format_move(player: str, move: Move) -> str:
    return f"{player} 出 {move}{MOVE_EMOJI[move]}"


def build_move_prompt(agent_name: str, state: GameState) -> str:
    opponent = "bob" if agent_name == "alice" else "alice"
    return (
        f"=== 第 {state.round_num} 局 ===\n"
        f"当前比分：Alice {state.alice_wins} - Bob {state.bob_wins}（先赢 {WINS_TO_WIN} 局者胜）\n"
        f"上局回顾：{state.last_summary}\n\n"
        f"你是 {agent_name}，对手是 {opponent}。\n"
        f"请出拳，只回复一个词：石头、剪刀 或 布"
    )


def create_agents() -> tuple[AssistantAgent, AssistantAgent, list]:
    alice_client = create_model_client(ALICE_PRESET, thinking=False, temperature=0.8)
    bob_client = create_model_client(BOB_PRESET, temperature=0.8)

    base_system = (
        "你正在参与猜拳对战（石头剪刀布），三局两胜。\n"
        "每轮必须且只能回复一个词：石头、剪刀 或 布。\n"
        "不要加解释、标点或其他文字。可以根据比分调整策略。"
    )

    alice = AssistantAgent(
        name="alice",
        model_client=alice_client,
        description="猜拳选手 Alice（智谱 GLM）",
        system_message=f"你是 Alice，性格活泼，出拳果断。\n{base_system}",
    )
    bob = AssistantAgent(
        name="bob",
        model_client=bob_client,
        description="猜拳选手 Bob（DeepSeek）",
        system_message=f"你是 Bob，性格沉稳，善于观察对手规律。\n{base_system}",
    )
    return alice, bob, [alice_client, bob_client]


async def ask_move(
    agent: AssistantAgent,
    agent_name: str,
    state: GameState,
) -> Move:
    prompt = build_move_prompt(agent_name, state)
    for attempt in range(MAX_RETRIES + 1):
        result = await agent.run(task=prompt, output_task_messages=False)
        reply = parse_agent_reply(result, agent_name)
        print_agent_reply(agent_name, reply.thought, reply.output, show_thought=SHOW_THOUGHT)

        move = parse_move(reply.output)
        if move:
            return move

        prompt = (
            f"格式错误，你回复了「{reply.output}」。\n"
            f"请只回复一个词：石头、剪刀 或 布。"
        )
        if attempt < MAX_RETRIES:
            print(f"[裁判] {agent_name} 出拳无效，请重试（{attempt + 1}/{MAX_RETRIES}）\n")

    raise RuntimeError(f"{agent_name} 连续 {MAX_RETRIES + 1} 次未能给出有效出拳")


async def play_round(
    alice: AssistantAgent,
    bob: AssistantAgent,
    state: GameState,
) -> None:
    state.round_num += 1
    print(f"{'=' * 40}")
    print(f"第 {state.round_num} 局开始（比分 Alice {state.alice_wins} - Bob {state.bob_wins}）")
    print(f"{'=' * 40}\n")

    alice_move, bob_move = await asyncio.gather(
        ask_move(alice, "alice", state),
        ask_move(bob, "bob", state),
    )

    print(f"[揭晓] {format_move('Alice', alice_move)}  vs  {format_move('Bob', bob_move)}")

    winner = judge(alice_move, bob_move)
    if winner == "draw":
        state.last_summary = f"{format_move('Alice', alice_move)}，{format_move('Bob', bob_move)}，平局"
        print("[结果] 平局，比分不变\n")
        return

    if winner == "alice":
        state.alice_wins += 1
        state.last_summary = (
            f"{format_move('Alice', alice_move)}，{format_move('Bob', bob_move)}，Alice 获胜"
        )
        print(f"[结果] Alice 赢本局！比分 {state.alice_wins} - {state.bob_wins}\n")
    else:
        state.bob_wins += 1
        state.last_summary = (
            f"{format_move('Alice', alice_move)}，{format_move('Bob', bob_move)}，Bob 获胜"
        )
        print(f"[结果] Bob 赢本局！比分 {state.alice_wins} - {state.bob_wins}\n")


async def main() -> None:
    alice, bob, clients = create_agents()
    state = GameState()

    try:
        print("=== 猜拳对战：Alice vs Bob（三局两胜）===")
        print(f"Alice: {ALICE_PRESET}  |  Bob: {BOB_PRESET}")
        print()
        print(GAME_RULES)
        print()

        while state.alice_wins < WINS_TO_WIN and state.bob_wins < WINS_TO_WIN:
            await play_round(alice, bob, state)

        if state.alice_wins >= WINS_TO_WIN:
            print(f"🎉 比赛结束！Alice 以 {state.alice_wins}-{state.bob_wins} 获胜！")
        else:
            print(f"🎉 比赛结束！Bob 以 {state.bob_wins}-{state.alice_wins} 获胜！")
    finally:
        for client in clients:
            await client.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
