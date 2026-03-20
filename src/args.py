from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ReapArgs:
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})


@dataclass
class ModelArgs:
    model_name: str = field(
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        metadata={"help": "HF model id or local model path."},
    )
    num_experts_per_tok_override: int | None = field(
        default=None,
        metadata={"help": "Optional override for experts-per-token at inference time."},
    )


@dataclass
class EvalArgs:
    use_server: bool = field(default=True, metadata={"help": "Use vLLM server backend."})
    greedy: bool = field(default=True, metadata={"help": "Use greedy decoding."})
    temperature: float = field(default=0.7, metadata={"help": "Sampling temperature."})
    top_p: float = field(default=0.8, metadata={"help": "Top-p for sampling."})
    top_k: int = field(default=20, metadata={"help": "Top-k for sampling."})
    min_p: float = field(default=0.0, metadata={"help": "Min-p for sampling."})

    results_dir: str | None = field(
        default=None,
        metadata={"help": "Directory to save evaluation results."},
    )

    run_lm_eval: bool = field(default=True, metadata={"help": "Run lm-eval."})
    run_evalplus: bool = field(default=True, metadata={"help": "Run EvalPlus."})
    run_livecodebench: bool = field(default=True, metadata={"help": "Run LiveCodeBench."})
    run_wildbench: bool = field(default=False, metadata={"help": "Run WildBench."})
    run_math: bool = field(default=False, metadata={"help": "Run math benchmarks."})

    math_datasets: list[str] = field(
        default_factory=lambda: ["gsm8k", "math_500"],
        metadata={"help": "Math datasets for evalscope."},
    )
    eval_max_length: int | None = field(
        default=2048,
        metadata={"help": "Maximum new tokens to generate during evaluation."},
    )
    eval_model_max_length: int | None = field(
        default=4096,
        metadata={"help": "Context length cap for evaluation backends."},
    )

    lm_eval_tasks: list[str] = field(
        default_factory=lambda: [
            "winogrande",
            "arc_challenge",
            "arc_easy",
            "boolq",
            "hellaswag",
            "mmlu",
            "openbookqa",
            "rte",
        ],
        metadata={"help": "lm-eval task list."},
    )
    evalplus_tasks: list[str] = field(
        default_factory=lambda: ["mbpp", "humaneval"],
        metadata={"help": "EvalPlus task list."},
    )
    evalplus_batch_size: int | None = field(
        default=64,
        metadata={"help": "EvalPlus batch size."},
    )
    lm_eval_apply_chat_template: bool = field(
        default=False,
        metadata={"help": "Apply chat template for lm-eval."},
    )
    lm_eval_num_fewshot: int = field(
        default=0,
        metadata={"help": "Few-shot count for lm-eval."},
    )
    lm_eval_fewshot_as_multiturn: bool = field(
        default=False,
        metadata={"help": "Format lm-eval few-shot examples as multi-turn chat."},
    )

    server_log_file_name: str = field(
        default_factory=lambda: os.environ.get("SERVER_LOG_FILE_NAME", "server.log"),
        metadata={"help": "Log file for the vLLM server."},
    )
    vllm_port: int = field(default=8000, metadata={"help": "Port for vLLM serve."})
    parallel_tasks: int = field(
        default=32,
        metadata={"help": "Parallel task count for evaluation helpers."},
    )
    wildbench_num_threads: int | None = field(
        default=4,
        metadata={"help": "WildBench/HELM worker count."},
    )
