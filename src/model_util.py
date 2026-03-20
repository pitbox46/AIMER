from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


MODEL_ATTRS = {
    "Qwen3MoeForCausalLM": {"moe_block": "mlp"},
    "Ernie4_5_MoeForCausalLM": {"moe_block": "mlp"},
    "Ernie4_5_MoEForCausalLM": {"moe_block": "mlp"},
    "OlmoeForCausalLM": {"moe_block": "mlp"},
}


def get_moe(model, layer: int):
    model_name = model.__class__.__name__
    if model_name not in MODEL_ATTRS:
        raise KeyError(f"Unsupported model class for MoE lookup: {model_name}")
    moe_attr_name = MODEL_ATTRS[model_name]["moe_block"]
    return getattr(model.model.layers[layer], moe_attr_name)


def patched_model_map(model: str) -> str:
    patched = False
    model_name = model

    if model == "deepseek-ai/DeepSeek-V2-Lite-Chat":
        patched = True
        model_name = "artifacts/models/DeepSeek-V2-Lite-Chat"
    elif model == "baidu/ERNIE-4.5-21B-A3B-PT":
        patched = True
        model_name = "baidu/ERNIE-4.5-21B-A3B-PT"
    elif model == "zai-org/GLM-4.5-Air":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air"
    elif model == "zai-org/GLM-4.5-Air-FP8":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air-FP8"
    elif model == "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8":
        patched = True
        model_name = "artifacts/models/Qwen3-Coder-480B-A35B-Instruct-FP8"
    elif model == "allenai/OLMoE-1B-7B-0125":
        patched = True
        model_name = "allenai/OLMoE-1B-7B-0125"
    elif model == "allenai/OLMoE-1B-7B-0125-Instruct":
        patched = True
        model_name = "allenai/OLMoE-1B-7B-0125-Instruct"
    elif model == "RedHatAI/Qwen3-30B-A3B-FP8-dynamic":
        patched = True
        model_name = "RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
    elif model == "Qwen/Qwen3-30B-A3B-Instruct-2507":
        patched = True
        model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    if patched:
        logger.info("Using patched model for %s from: %s", model, model_name)
    return model_name
