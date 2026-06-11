"""模型客户端工厂：统一创建 GLM、DeepSeek 等 OpenAI 兼容模型的 AutoGen 客户端。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import certifi
import httpx
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

Provider = Literal["glm", "deepseek", "openai"]
ThinkingMode = Literal["enabled", "disabled"]


@dataclass(frozen=True)
class ModelPreset:
    provider: Provider
    model: str
    base_url: str
    api_key_env: tuple[str, ...]
    vision: bool = False
    function_calling: bool = True
    json_output: bool = True
    structured_output: bool = True
    # 智谱等 OpenAI 兼容接口通常不支持 messages.name 字段
    include_name_in_message: bool = True
    add_name_prefixes: bool = False


MODEL_PRESETS: dict[str, ModelPreset] = {
    # 智谱 GLM（Coding Plan）
    "glm-5": ModelPreset(
        provider="glm",
        model="glm-5.1",
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        api_key_env=("ZHIPUAI_API_KEY", "ZAI_API_KEY"),
        include_name_in_message=False,
        add_name_prefixes=True,
    ),
    "glm-4.7": ModelPreset(
        provider="glm",
        model="glm-4.7",
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        api_key_env=("ZHIPUAI_API_KEY", "ZAI_API_KEY"),
        include_name_in_message=False,
        add_name_prefixes=True,
    ),
    # 智谱 GLM（按量付费）
    "glm-5-pay": ModelPreset(
        provider="glm",
        model="glm-5.1",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env=("ZHIPUAI_API_KEY", "ZAI_API_KEY"),
        include_name_in_message=False,
        add_name_prefixes=True,
    ),
    # DeepSeek
    "deepseek-v4-flash": ModelPreset(
        provider="deepseek",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key_env=("DEEPSEEK_API_KEY",),
    ),
    # OpenAI（可选）
    "gpt-4o-mini": ModelPreset(
        provider="openai",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env=("OPENAI_API_KEY",),
        vision=True,
    ),
}


def create_http_client(*, timeout: float = 60.0) -> httpx.AsyncClient:
    """创建 HTTP 客户端，绕过系统代理以避免 SSL 证书校验失败。"""
    return httpx.AsyncClient(
        verify=certifi.where(),
        trust_env=False,
        timeout=timeout,
    )


def _resolve_api_key(preset: ModelPreset, api_key: str | None) -> str:
    if api_key:
        return api_key
    for env_name in preset.api_key_env:
        value = os.getenv(env_name)
        if value:
            return value
    env_hint = " 或 ".join(preset.api_key_env)
    raise RuntimeError(
        f"未找到 {preset.provider} 模型的 API Key，请设置环境变量: {env_hint}\n"
        f"PowerShell 示例: $env:{preset.api_key_env[0]} = 'your-api-key'"
    )


def _build_model_info(preset: ModelPreset) -> ModelInfo | None:
    if preset.provider == "openai":
        return None
    return {
        "vision": preset.vision,
        "function_calling": preset.function_calling,
        "json_output": preset.json_output,
        "family": ModelFamily.UNKNOWN,
        "structured_output": preset.structured_output,
    }


def _resolve_thinking_mode(thinking: bool | ThinkingMode | None) -> ThinkingMode | None:
    if thinking is None:
        return None
    if isinstance(thinking, bool):
        return "enabled" if thinking else "disabled"
    return thinking


def _build_glm_extra_body(thinking: bool | ThinkingMode | None) -> dict[str, object] | None:
    """构造智谱 GLM 的 thinking 参数（通过 extra_body 传给 API）。"""
    mode = _resolve_thinking_mode(thinking)
    if mode is None:
        return None
    return {"thinking": {"type": mode}}


def create_model_client(
    preset_name: str,
    *,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    thinking: bool | ThinkingMode | None = None,
) -> OpenAIChatCompletionClient:
    """根据预设名称创建模型客户端。

    Args:
        preset_name: 预设名称，如 "glm-5"、"deepseek-chat"。
        api_key: 可选，手动传入 API Key（默认从环境变量读取）。
        temperature: 采样温度。
        max_tokens: 最大输出 token 数。
        timeout: HTTP 请求超时（秒）。
        thinking: 仅对智谱 GLM 模型生效。True/"enabled" 开启深度思考，
            False/"disabled" 关闭，None 不传（使用 API 默认行为）。

    Returns:
        配置好的 OpenAIChatCompletionClient 实例。
    """
    if preset_name not in MODEL_PRESETS:
        available = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"未知模型预设: {preset_name}，可选: {available}")

    preset = MODEL_PRESETS[preset_name]
    client_kwargs: dict = {
        "model": preset.model,
        "api_key": _resolve_api_key(preset, api_key),
        "http_client": create_http_client(timeout=timeout),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if preset.provider != "openai":
        client_kwargs["base_url"] = preset.base_url
        model_info = _build_model_info(preset)
        if model_info is not None:
            client_kwargs["model_info"] = model_info
        client_kwargs["include_name_in_message"] = preset.include_name_in_message
        client_kwargs["add_name_prefixes"] = preset.add_name_prefixes

    if preset.provider == "glm":
        extra_body = _build_glm_extra_body(thinking)
        if extra_body is not None:
            client_kwargs["extra_body"] = extra_body

    return OpenAIChatCompletionClient(**client_kwargs)


def list_presets() -> list[str]:
    """返回所有可用的模型预设名称。"""
    return sorted(MODEL_PRESETS)
