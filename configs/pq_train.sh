#!/usr/bin/env bash
set -euo pipefail

# Usage: bash ./configs/pq_train.sh
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark="pq"
graph_regime="${GRAPH_REGIME:-filtered}"
if [[ "${graph_regime}" != "filtered" && "${graph_regime}" != "source" ]]; then
    echo "GRAPH_REGIME must be 'filtered' or 'source'." >&2
    exit 1
fi
dataset="${benchmark}_${graph_regime}"
seed="${SEED:-42}"
epochs="${EPOCHS:-100}"
device="${DEVICE:-auto}"
checkpoint_dir="${PQ_CHECKPOINT_DIR:-${repo_root}/saved_model/${dataset}_seed_${seed}}"

if [[ "${checkpoint_dir}" != /* ]]; then
    checkpoint_dir="${repo_root}/${checkpoint_dir}"
fi

if [[ ! -f "${repo_root}/data/THESEUS/${dataset}/train_data.txt" ]]; then
    echo "Prepared ${dataset} data not found. First run:" >&2
    echo "  python script/preprocess/prepare_pathquestion.py --dataset pq --graph-regime ${graph_regime} --reconstruction-root /path/to/PathQuestion-Family/generated" >&2
    exit 1
fi

if [[ -e "${checkpoint_dir}/model.pth" ]]; then
    echo "Checkpoint already exists: ${checkpoint_dir}/model.pth" >&2
    echo "Set PQ_CHECKPOINT_DIR to a new directory to preserve this run." >&2
    exit 1
fi

mkdir -p "${checkpoint_dir}"
export PYTHONHASHSEED="${seed}"

# UHop training paths are relative to src.
cd "${repo_root}/src"
python main.py \
    --train \
    --framework UHop \
    --model HR_BiLSTM_plus \
    --dataset "${dataset}" \
    --emb_size 300 \
    --hidden_size 100 \
    --dynamic recurrent \
    --epoch_num "${epochs}" \
    --seed "${seed}" \
    --device "${device}" \
    --path "${checkpoint_dir}"
