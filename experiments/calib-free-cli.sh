#!/bin/bash
set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES=${1}
FIRST_DEVICE=$(echo "$1" | cut -d',' -f1)
port=$((8300 + FIRST_DEVICE))

model_name=${2:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}
metric_spec=${3:-"aimer"}
sparsity_ratio=${4:-0.25}
seed=${5:-42}

case "${metric_spec,,}" in
  aimer)
    metric_spec="aimer"
    ;;
esac

# qa
run_lm_eval=${6:-true}
# coding
run_evalplus=${7:-true}
run_livecodebench=${8:-true}
# math
run_math=${9:-true}
# wildbench
run_wildbench=${10:-true}

lm_eval_apply_chat_template=${11:-false}
lm_eval_num_fewshot=${12:-0}
lm_eval_fewshot_as_multiturn=${13:-false}
use_server=${14:-true}
output_dir_override=${15:-""}
run_final_eval=${16:-true}
local_files_only=${17:-true}
trust_remote_code=${18:-true}
direction=${19:-"auto"}
device_map=${20:-"auto"}
force_prune=${22:-false}
force_prune_norm="${force_prune,,}"
metric_device=${23:-"auto"}
uniform_pruning=${24:-true}
uniform_pruning_norm="${uniform_pruning,,}"

if [[ "${uniform_pruning_norm}" != "true" ]]; then
  echo "Calibration-free pruning now supports only uniform pruning; overriding to true."
  uniform_pruning=true
  uniform_pruning_norm="true"
fi

ratio_fmt=$(python - <<PY
ratio = float("${sparsity_ratio}")
print(f"{ratio:.3f}")
PY
)

short_model_name=$(echo "${model_name}" | awk -F/ '{print $NF}')
metric_tag=$(python - <<PY
import re
spec = """${metric_spec}"""
tag = re.sub(r"[^A-Za-z0-9._-]+", "_", spec.strip())
tag = re.sub(r"_+", "_", tag).strip("._-")
print(tag or "metric")
PY
)

if [[ -n "${output_dir_override}" ]]; then
  model_dir="${output_dir_override}"
else
  uniform_suffix=""
  if [[ "${uniform_pruning_norm}" == "true" ]]; then
    uniform_suffix="_uniform"
  fi
  model_dir="artifacts/calib-free/${short_model_name}/${metric_tag}_${ratio_fmt}${uniform_suffix}"
fi

mkdir -p "${model_dir}"
prune_log="${model_dir}/calib-free-prune-${FIRST_DEVICE}.log"
eval_log="${model_dir}/calib-free-eval-${FIRST_DEVICE}.log"

echo "Running calibration-free pruning"
echo "  model_name=${model_name}"
echo "  metric_spec=${metric_spec}"
echo "  sparsity_ratio=${ratio_fmt}"
echo "  output_dir=${model_dir}"
echo "  prune_log=${prune_log}"
echo "  force_prune=${force_prune_norm}"
echo "  metric_device=${metric_device}"
echo "  uniform_pruning=${uniform_pruning_norm}"

have_pruned_json=false
have_config=false
have_weights=false
if [[ -f "${model_dir}/pruned_experts.json" ]]; then
  have_pruned_json=true
fi
if [[ -f "${model_dir}/config.json" ]]; then
  have_config=true
fi
if compgen -G "${model_dir}/*.safetensors" > /dev/null; then
  have_weights=true
elif [[ -f "${model_dir}/pytorch_model.bin" ]]; then
  have_weights=true
elif compgen -G "${model_dir}/pytorch_model-*.bin" > /dev/null; then
  have_weights=true
fi

model_exists=false
if [[ "${have_pruned_json}" == "true" && "${have_config}" == "true" && "${have_weights}" == "true" ]]; then
  model_exists=true
fi

if [[ "${model_exists}" == "true" && "${force_prune_norm}" != "true" ]]; then
  echo "Found existing pruned model artifacts in ${model_dir}; skipping pruning."
else
  python src/reap/calib_free_prune.py \
    --model-name "${model_name}" \
    --metric "${metric_spec}" \
    --sparsity-ratio "${sparsity_ratio}" \
    --seed "${seed}" \
    --output-dir "${model_dir}" \
    --device-map "${device_map}" \
    --torch-dtype "bfloat16" \
    --metric-device "${metric_device}" \
    --uniform-pruning "${uniform_pruning}" \
    --local-files-only "${local_files_only}" \
    --trust-remote-code "${trust_remote_code}" | tee "${prune_log}"
fi

if [[ "${run_final_eval}" != "true" ]]; then
  echo "Skipping evaluation. Model directory: ${model_dir}"
  exit 0
fi

echo "Evaluating model: ${model_dir}"
bash experiments/eval.sh \
  "${model_dir}" \
  "${seed}" \
  "${port}" \
  "${eval_log}" \
  "${run_lm_eval}" \
  "${run_evalplus}" \
  "${run_livecodebench}" \
  "${run_math}" \
  "${run_wildbench}" \
  "${lm_eval_apply_chat_template}" \
  "${lm_eval_num_fewshot}" \
  "${lm_eval_fewshot_as_multiturn}" \
  "" \
  "${use_server}"

echo "Finished evaluating model: ${model_dir}"
