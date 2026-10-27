#!/usr/bin/env bash
set -euo pipefail

# Usage: bash ./configs/mquake_st_multi_eval.sh
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="mquake_st_multi"
seed="${SEED:-42}"
device="${DEVICE:-auto}"
checkpoint_dir="${MQUAKE_ST_MULTI_CHECKPOINT_DIR:-${repo_root}/saved_model/${dataset}_seed_${seed}}"

if [[ "${checkpoint_dir}" != /* ]]; then
    checkpoint_dir="${repo_root}/${checkpoint_dir}"
fi

if [[ ! -f "${checkpoint_dir}/model.pth" || ! -f "${checkpoint_dir}/args.txt" ]]; then
    echo "Missing checkpoint in ${checkpoint_dir}." >&2
    echo "Run bash ./configs/mquake_st_multi_train.sh first, or set MQUAKE_ST_MULTI_CHECKPOINT_DIR." >&2
    exit 1
fi

export PYTHONHASHSEED="${seed}"
cd "${repo_root}"
python script/evaluate_theseus.py \
    --checkpoint "${checkpoint_dir}" \
    --split test \
    --max-hops 4 \
    --question-variant canonical \
    --entity-selector uniform \
    --termination-mode learned \
    --seed "${seed}" \
    --device "${device}" \
    "$@"
