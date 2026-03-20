#!/usr/bin/env bash
set -euo pipefail

update_submodule() {
    local path="$1"
    git submodule update --init --recursive --checkout "$path"
}

update_helm_submodule() {
    if update_submodule "third-party/helm"; then
        return 0
    fi

    echo "Pinned third-party/helm commit is unavailable; falling back to origin/main." >&2
    git -C third-party/helm fetch origin main
    git -C third-party/helm checkout --force FETCH_HEAD
}

git submodule init
git submodule update
git submodule sync --recursive
update_submodule "third-party/evalplus"
update_submodule "third-party/llm-compressor"
update_submodule "third-party/LiveCodeBench"
update_helm_submodule
update_submodule "third-party/creative-writing-bench"
update_submodule "third-party/evalscope"
uv venv .venv --seed --python 3.12
uv pip install --upgrade pip
uv pip install setuptools wheel  # --seed not working in some cases
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -vv --torch-backend auto
