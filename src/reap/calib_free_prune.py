from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import random
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reap.model_util import patched_model_map

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SUPPORTED_AIMER_ALIASES = {"aimer"}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_metric_tag(metric_spec: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", metric_spec.strip())
    tag = re.sub(r"_+", "_", tag).strip("._-")
    return tag or "metric"


def normalize_metric_name(metric: str) -> str:
    normalized = str(metric).strip().lower()
    if normalized in SUPPORTED_AIMER_ALIASES:
        return "aimer"
    raise ValueError(
        "Calibration-free pruning now supports only the AIMER metric. "
        f"Received {metric!r}."
    )


def _resolve_torch_dtype(dtype_name: str):
    name = dtype_name.strip().lower()
    if name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported --torch-dtype '{dtype_name}'.")
    return mapping[name]


def _as_metric_f32(tensor: torch.Tensor, metric_device: str) -> torch.Tensor:
    t = tensor.detach()
    if metric_device == "cpu":
        return t.to(dtype=torch.float32, device="cpu")
    if metric_device == "auto":
        return t.to(dtype=torch.float32)
    raise ValueError(f"Unsupported metric_device={metric_device!r}; expected 'auto' or 'cpu'.")


def _is_ernie_moe_layer(config, layer_idx: int) -> bool:
    try:
        end_idx = int(config.moe_layer_end_index)
        if end_idx == -1:
            end_idx = int(config.num_hidden_layers) - 1
        return (
            ((layer_idx + 1) % int(config.moe_layer_interval) == 0)
            and layer_idx >= int(config.moe_layer_start_index)
            and layer_idx <= end_idx
        )
    except Exception:
        return False


def _is_qwen3_moe_layer(config, layer_idx: int) -> bool:
    try:
        mlp_only_layers = getattr(config, "mlp_only_layers", None) or []
        if layer_idx in mlp_only_layers:
            return False
        decoder_sparse_step = int(getattr(config, "decoder_sparse_step", 1))
        num_experts = getattr(config, "num_experts", 0)
        get_num_experts = getattr(config, "get_num_experts", None)
        if callable(get_num_experts):
            num_experts = get_num_experts(layer_idx)
        return int(num_experts) > 0 and ((layer_idx + 1) % decoder_sparse_step == 0)
    except Exception:
        return False


def _iter_moe_layers(model: torch.nn.Module) -> list[tuple[int, torch.nn.Module, int]]:
    model_type = getattr(model.config, "model_type", "")
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    items: list[tuple[int, torch.nn.Module, int]] = []

    for layer_idx in range(num_layers):
        if model_type == "ernie4_5_moe" and not _is_ernie_moe_layer(model.config, layer_idx):
            continue
        if model_type == "qwen3_moe" and not _is_qwen3_moe_layer(model.config, layer_idx):
            continue

        layer = model.model.layers[layer_idx]
        moe = getattr(layer, "mlp", None)
        if moe is None:
            continue
        gate = getattr(moe, "gate", None)
        weight = getattr(gate, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue

        num_experts = int(weight.shape[0])
        min_keep = 1
        if model_type == "ernie4_5_moe":
            min_keep = int(getattr(model.config, "moe_k", min_keep))
        elif hasattr(moe, "top_k"):
            min_keep = int(getattr(moe, "top_k"))
        elif hasattr(model.config, "num_experts_per_tok"):
            min_keep = int(getattr(model.config, "num_experts_per_tok"))

        min_keep = max(1, min(num_experts, min_keep))
        items.append((layer_idx, moe, min_keep))

    return items


def _extract_projection_tensors(
    moe: torch.nn.Module,
    expert_id: int,
    metric_device: str,
) -> dict[str, torch.Tensor]:
    experts = getattr(moe, "experts", None)
    out: dict[str, torch.Tensor] = {}

    if isinstance(experts, torch.nn.ModuleList):
        exp = experts[expert_id]
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj_module = getattr(exp, proj_name, None)
            weight = getattr(proj_module, "weight", None)
            if isinstance(weight, torch.Tensor):
                out[proj_name] = _as_metric_f32(weight, metric_device)
        return out

    gate_up_proj = getattr(experts, "gate_up_proj", None)
    down_proj = getattr(experts, "down_proj", None)
    if not isinstance(gate_up_proj, torch.Tensor) or not isinstance(down_proj, torch.Tensor):
        return out

    gate_up = _as_metric_f32(gate_up_proj[expert_id], metric_device)
    down = _as_metric_f32(down_proj[expert_id], metric_device)

    if gate_up.ndim == 2 and gate_up.shape[0] >= 2 and gate_up.shape[0] % 2 == 0:
        split = int(gate_up.shape[0] // 2)
        gate = gate_up[:split]
        up = gate_up[split:]
    else:
        gate = gate_up
        up = gate_up

    out["gate_proj"] = gate
    out["up_proj"] = up
    out["down_proj"] = down
    return out


def get_proj_weights(
    layer: torch.nn.Module,
    expert_id: int,
    metric_device: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projections = _extract_projection_tensors(layer, expert_id, metric_device=metric_device)
    gate = projections.get("gate_proj")
    up = projections.get("up_proj")
    down = projections.get("down_proj")
    if gate is None or up is None or down is None:
        raise ValueError(f"Failed to extract all projection weights for expert {expert_id}.")
    return gate, up, down


def _aimer_scores_and_rank(
    layer: torch.nn.Module,
    metric_device: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = []
    num_experts = int(layer.gate.weight.shape[0])

    for expert_id in range(num_experts):
        gate, up, down = get_proj_weights(layer, expert_id, metric_device=metric_device)
        abs_sum = gate.abs().sum() + up.abs().sum() + down.abs().sum()
        numel = gate.numel() + up.numel() + down.numel()
        l2_sq = gate.square().sum() + up.square().sum() + down.square().sum()
        if numel <= 0 or float(l2_sq.item()) <= 0.0:
            score = torch.zeros((), dtype=torch.float32, device=gate.device)
        else:
            score = (abs_sum / numel) / torch.sqrt(l2_sq / numel)
        scores.append(score.to(dtype=torch.float32))

    stacked_scores = torch.stack(scores)
    _, sorted_idx = torch.sort(stacked_scores, descending=True)
    return stacked_scores.to(device="cpu"), sorted_idx.to(device="cpu")


def aimer_rank(layer: torch.nn.Module, metric_device: str = "auto") -> torch.Tensor:
    _, sorted_idx = _aimer_scores_and_rank(layer, metric_device=metric_device)
    return sorted_idx


def _allowed_prune_counts(
    *,
    total_experts: int,
    max_prune: int,
) -> list[int]:
    _ = total_experts
    return list(range(0, max_prune + 1))


def _select_uniform_pruned_experts(
    ranked_experts_by_layer: dict[int, list[int]],
    layer_total_experts: dict[int, int],
    layer_max_prune: dict[int, int],
    sparsity_ratio: float,
) -> tuple[dict[int, list[int]], int, int, int]:
    layers = sorted(ranked_experts_by_layer)
    if not layers:
        raise ValueError("No MoE layers were detected for pruning.")

    total_experts = sum(int(layer_total_experts[layer]) for layer in layers)
    target_pruned = int(round(total_experts * float(sparsity_ratio)))
    target_pruned = max(0, min(total_experts, target_pruned))

    common_allowed: set[int] | None = None
    for layer in layers:
        allowed = set(
            _allowed_prune_counts(
                total_experts=int(layer_total_experts[layer]),
                max_prune=int(layer_max_prune[layer]),
            )
        )
        if not allowed:
            raise ValueError(
                f"Layer {layer} has no feasible prune counts under the current constraints."
            )
        common_allowed = allowed if common_allowed is None else (common_allowed & allowed)

    feasible_k = sorted(common_allowed) if common_allowed else []
    if not feasible_k:
        raise ValueError(
            "Uniform pruning is infeasible: no shared per-layer prune count satisfies "
            "all constraints across layers."
        )

    feasible_totals = [(k, int(k) * len(layers)) for k in feasible_k]
    min_feasible = min(total for _, total in feasible_totals)
    max_feasible = max(total for _, total in feasible_totals)

    if target_pruned < min_feasible:
        logger.warning(
            "Requested prune count %d is below min feasible %d under uniform pruning; clipping upward.",
            target_pruned,
            min_feasible,
        )
        target_pruned = min_feasible
    if target_pruned > max_feasible:
        logger.warning(
            "Requested prune count %d exceeds max feasible %d under uniform pruning; clipping.",
            target_pruned,
            max_feasible,
        )
        target_pruned = max_feasible

    exact = [k for k, total in feasible_totals if total == target_pruned]
    if exact:
        uniform_k = int(exact[0])
        selected_pruned = int(target_pruned)
    else:
        uniform_k, selected_pruned = min(
            feasible_totals,
            key=lambda item: (
                abs(item[1] - target_pruned),
                0 if item[1] <= target_pruned else 1,
                item[1],
            ),
        )
        uniform_k = int(uniform_k)
        logger.warning(
            "Target prune count %d is not exactly reachable under uniform pruning; "
            "using closest feasible count %d (k=%d per layer).",
            target_pruned,
            selected_pruned,
            uniform_k,
        )

    pruned: dict[int, list[int]] = {}
    for layer in layers:
        if uniform_k <= 0:
            continue
        pruned[layer] = sorted(int(expert_id) for expert_id in ranked_experts_by_layer[layer][:uniform_k])

    return pruned, target_pruned, int(selected_pruned), uniform_k


def _slice_last_dim_parameter(param: torch.nn.Parameter, retain_indices: list[int]) -> torch.nn.Parameter:
    sliced = param.data.index_select(
        dim=param.ndim - 1,
        index=torch.tensor(retain_indices, device=param.device, dtype=torch.long),
    ).clone()
    return torch.nn.Parameter(sliced, requires_grad=param.requires_grad)


def _prune_moe_layer_in_place(moe: torch.nn.Module, pruned_experts: list[int]) -> int:
    gate = getattr(moe, "gate", None)
    if gate is None or not isinstance(getattr(gate, "weight", None), torch.Tensor):
        raise ValueError("MoE layer is missing a valid gate.weight tensor.")

    num_experts = int(gate.weight.shape[0])
    retain_indices = sorted(i for i in range(num_experts) if i not in pruned_experts)
    kept = len(retain_indices)
    if kept <= 0:
        raise ValueError(f"Cannot prune all experts from a layer (num_experts={num_experts}).")

    index = torch.tensor(retain_indices, device=gate.weight.device, dtype=torch.long)

    with torch.no_grad():
        gate.weight = torch.nn.Parameter(gate.weight.data.index_select(0, index).clone())
        if getattr(gate, "bias", None) is not None:
            gate.bias = torch.nn.Parameter(gate.bias.data.index_select(0, index).clone())

        moe_statics = getattr(gate, "moe_statics", None)
        correction_bias = getattr(moe_statics, "e_score_correction_bias", None)
        if isinstance(correction_bias, torch.nn.Parameter) and correction_bias.ndim >= 1:
            if correction_bias.shape[-1] != num_experts:
                raise ValueError(
                    "Unexpected Ernie correction-bias shape while pruning experts: "
                    f"{tuple(correction_bias.shape)}"
                )
            moe_statics.e_score_correction_bias = _slice_last_dim_parameter(
                correction_bias,
                retain_indices,
            )

        experts = getattr(moe, "experts", None)
        if isinstance(experts, torch.nn.ModuleList):
            moe.experts = torch.nn.ModuleList([experts[i] for i in retain_indices])
        elif experts is not None:
            gate_up_proj = getattr(experts, "gate_up_proj", None)
            down_proj = getattr(experts, "down_proj", None)
            if isinstance(gate_up_proj, torch.Tensor):
                experts.gate_up_proj = torch.nn.Parameter(gate_up_proj.data.index_select(0, index).clone())
            if isinstance(down_proj, torch.Tensor):
                experts.down_proj = torch.nn.Parameter(down_proj.data.index_select(0, index).clone())
            if hasattr(experts, "num_experts"):
                experts.num_experts = kept

    if hasattr(moe, "num_experts"):
        moe.num_experts = kept
    if hasattr(moe, "top_k"):
        moe.top_k = min(int(moe.top_k), kept)
    if hasattr(gate, "num_experts"):
        gate.num_experts = kept
    if hasattr(gate, "top_k"):
        gate.top_k = min(int(gate.top_k), kept)

    return kept


def _update_model_config_after_uniform_pruning(model: torch.nn.Module, kept_per_layer: list[int]) -> None:
    if not kept_per_layer:
        return

    unique_counts = sorted(set(int(value) for value in kept_per_layer))
    if len(unique_counts) != 1:
        raise ValueError(
            "Uniform pruning expected a single retained-expert count across MoE layers, "
            f"but found {unique_counts}."
        )

    kept = int(unique_counts[0])
    model_type = getattr(model.config, "model_type", "")

    if model_type == "olmoe":
        model.config.num_experts = kept
        if hasattr(model.config, "num_experts_per_tok"):
            model.config.num_experts_per_tok = min(int(model.config.num_experts_per_tok), kept)
    elif model_type == "ernie4_5_moe":
        model.config.moe_num_experts = kept
        if hasattr(model.config, "moe_k"):
            model.config.moe_k = min(int(model.config.moe_k), kept)
    elif model_type == "qwen3_moe":
        model.config.num_experts = kept
        if hasattr(model.config, "num_experts_per_tok"):
            model.config.num_experts_per_tok = min(int(model.config.num_experts_per_tok), kept)
    else:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. Supported: olmoe, ernie4_5_moe, qwen3_moe."
        )


def _default_output_dir(
    model_name: str,
    metric_spec: str,
    ratio: float,
    uniform_pruning: bool,
) -> pathlib.Path:
    short_model = pathlib.Path(model_name).name
    metric_tag = sanitize_metric_tag(metric_spec)
    suffix = ""
    if uniform_pruning:
        suffix += "_uniform"
    return pathlib.Path("artifacts") / "calib-free" / short_model / f"{metric_tag}_{ratio:.3f}{suffix}"


def _write_scores_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_score_table_rows(
    records_by_layer: dict[int, list[dict[str, Any]]],
    pruned: dict[int, list[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_idx in sorted(records_by_layer):
        records = records_by_layer[layer_idx]
        pruned_set = set(pruned.get(layer_idx, []))
        ranked = sorted(
            records,
            key=lambda rec: (-float(rec["calib_free_score"]), int(rec["expert_id"])),
        )
        for rank, rec in enumerate(ranked, start=1):
            expert_id = int(rec["expert_id"])
            rows.append(
                {
                    "layer": int(layer_idx),
                    "expert_id": expert_id,
                    "aimer_score": float(rec["aimer_score"]),
                    "calib_free_score": float(rec["calib_free_score"]),
                    "score_rank_in_layer": int(rank),
                    "is_pruned": int(expert_id in pruned_set),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibration-free uniform MoE pruning with the AIMER weight-only metric."
    )
    parser.add_argument("--model-name", required=True, help="HF model id or local model path.")
    parser.add_argument(
        "--metric",
        default="aimer",
        help=(
            "Only 'aimer' is supported."
        ),
    )
    parser.add_argument(
        "--sparsity-ratio",
        type=float,
        required=True,
        help="Global prune ratio over experts in MoE layers.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="", help="Override output directory.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map (default: auto).")
    parser.add_argument(
        "--metric-device",
        choices=["auto", "cpu"],
        default="auto",
        help="Device used for metric extraction: auto=keep tensor on its native device, cpu=move to CPU.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", help="auto|float16|bfloat16|float32")
    parser.add_argument("--trust-remote-code", type=str2bool, default=True)
    parser.add_argument("--local-files-only", type=str2bool, default=True)
    parser.add_argument("--save-score-csv", type=str2bool, default=True)
    parser.add_argument(
        "--uniform-pruning",
        type=str2bool,
        default=True,
        help="Kept for CLI compatibility. Calibration-free pruning now always uses uniform pruning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sparsity_ratio < 0 or args.sparsity_ratio > 1:
        raise ValueError("--sparsity-ratio must be in [0, 1].")

    metric_name = normalize_metric_name(args.metric)
    if not bool(args.uniform_pruning):
        raise ValueError("Calibration-free pruning now supports only uniform pruning.")

    set_seed(args.seed)

    model_name = patched_model_map(args.model_name)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else _default_output_dir(
        args.model_name,
        metric_name,
        float(args.sparsity_ratio),
        True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )

    logger.info("Loading model from %s", model_name)
    device_map: Any = None if str(args.device_map).strip().lower() == "none" else args.device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=_resolve_torch_dtype(args.torch_dtype),
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )

    model_type = getattr(model.config, "model_type", "")
    if model_type not in {"olmoe", "ernie4_5_moe", "qwen3_moe"}:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. Supported: olmoe, ernie4_5_moe, qwen3_moe."
        )

    moe_layers = _iter_moe_layers(model)
    if not moe_layers:
        raise ValueError("No MoE layers were detected for calibration-free pruning.")

    logger.info(
        "Detected %d MoE layers for pruning (%s).",
        len(moe_layers),
        ", ".join(str(layer_idx) for layer_idx, _, _ in moe_layers),
    )

    records_by_layer: dict[int, list[dict[str, Any]]] = {}
    ranked_experts_by_layer: dict[int, list[int]] = {}
    layer_total_experts: dict[int, int] = {}
    layer_max_prune: dict[int, int] = {}

    for layer_idx, moe, min_keep in moe_layers:
        scores, sorted_idx = _aimer_scores_and_rank(moe, metric_device=str(args.metric_device))
        num_experts = int(scores.shape[0])
        ranked_experts_by_layer[layer_idx] = [int(expert_id) for expert_id in sorted_idx.tolist()]
        layer_total_experts[layer_idx] = num_experts
        layer_max_prune[layer_idx] = max(0, num_experts - int(min_keep))
        records_by_layer[layer_idx] = [
            {
                "layer": int(layer_idx),
                "expert_id": int(expert_id),
                "aimer_score": float(scores[expert_id].item()),
                "calib_free_score": float(scores[expert_id].item()),
            }
            for expert_id in range(num_experts)
        ]

    pruned, target_pruned, selected_pruned, uniform_k = _select_uniform_pruned_experts(
        ranked_experts_by_layer=ranked_experts_by_layer,
        layer_total_experts=layer_total_experts,
        layer_max_prune=layer_max_prune,
        sparsity_ratio=float(args.sparsity_ratio),
    )

    kept_per_layer: list[int] = []
    for layer_idx, moe, _min_keep in moe_layers:
        kept = _prune_moe_layer_in_place(moe, pruned.get(layer_idx, []))
        kept_per_layer.append(int(kept))

    _update_model_config_after_uniform_pruning(model, kept_per_layer)

    logger.info(
        "Pruning experts selected: target=%d selected=%d",
        target_pruned,
        selected_pruned,
    )
    logger.info("Uniform pruning enabled: k=%d pruned experts per layer.", uniform_k)

    logger.info("Saving pruned model to %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    plan_rows: list[dict[str, Any]] = []
    for layer_idx in sorted(records_by_layer):
        total = len(records_by_layer[layer_idx])
        pruned_count = len(pruned.get(layer_idx, []))
        plan_rows.append(
            {
                "layer": int(layer_idx),
                "total_experts": int(total),
                "pruned_experts": int(pruned_count),
                "kept_experts": int(total - pruned_count),
                "prune_ratio": (float(pruned_count) / float(total) if total > 0 else 0.0),
            }
        )

    pruned_experts_info_str = {
        str(layer): sorted(list(experts))
        for layer, experts in sorted(pruned.items())
        if experts
    }
    with open(output_dir / "pruned_experts.json", "w") as f:
        json.dump(pruned_experts_info_str, f, indent=2)

    with open(output_dir / "pruning_plan.json", "w") as f:
        json.dump(
            {
                "plan": plan_rows,
                "metric": metric_name,
                "uniform_pruning": True,
                "uniform_pruned_per_layer": int(uniform_k),
            },
            f,
            indent=2,
        )

    metadata = {
        "model_name": args.model_name,
        "patched_model_name": model_name,
        "model_type": model_type,
        "metric_input": args.metric,
        "metric_spec": metric_name,
        "metric_name": metric_name,
        "sparsity_ratio": float(args.sparsity_ratio),
        "target_pruned": int(target_pruned),
        "selected_pruned": int(selected_pruned),
        "total_moe_experts": int(sum(len(v) for v in records_by_layer.values())),
        "metric_device": str(args.metric_device),
        "uniform_pruning": True,
        "uniform_pruned_per_layer": int(uniform_k),
        "seed": int(args.seed),
    }
    with open(output_dir / "calib_free_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if bool(args.save_score_csv):
        rows_for_csv: list[dict[str, Any]] = []
        for layer_idx in sorted(records_by_layer):
            pruned_set = set(pruned.get(layer_idx, []))
            for rec in sorted(records_by_layer[layer_idx], key=lambda x: int(x["expert_id"])):
                row = dict(rec)
                row["is_pruned"] = int(int(row["expert_id"]) in pruned_set)
                rows_for_csv.append(row)
        _write_scores_csv(output_dir / "calib_free_scores.csv", rows_for_csv)

        score_table_rows = _build_score_table_rows(
            records_by_layer=records_by_layer,
            pruned=pruned,
        )
        _write_scores_csv(output_dir / "calib_free_score_table.csv", score_table_rows)

    logger.info("Saved calibration-free pruned model at %s", output_dir)
    logger.info("PRUNED_MODEL_DIR=%s", output_dir)
    print(f"PRUNED_MODEL_DIR={output_dir}")


if __name__ == "__main__":
    main()
