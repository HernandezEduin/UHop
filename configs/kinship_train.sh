#!/usr/bin/env bash
set -euo pipefail

# Usage: bash ./configs/kinship_train.sh
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
seed="${SEED:-42}"
epochs="${EPOCHS:-100}"
device="${DEVICE:-auto}"
checkpoint_dir="${KINSHIP_CHECKPOINT_DIR:-${repo_root}/saved_model/kinship_seed_${seed}}"

if [[ "${checkpoint_dir}" != /* ]]; then
    checkpoint_dir="${repo_root}/${checkpoint_dir}"
fi

if [[ ! -f "${repo_root}/data/THESEUS/kinship/train_data.txt" ]]; then
    echo "Kinship data not found. First run:" >&2
    echo "  python script/preprocess/prepare_theseus.py --dataset kinship" >&2
    exit 1
fi

if [[ -e "${checkpoint_dir}/model.pth" ]]; then
    echo "Checkpoint already exists: ${checkpoint_dir}/model.pth" >&2
    echo "Set KINSHIP_CHECKPOINT_DIR to a new directory to preserve this run." >&2
    exit 1
fi

mkdir -p "${checkpoint_dir}"
export PYTHONHASHSEED="${seed}"

# main.py expects paths relative to src.
cd "${repo_root}/src"
python main.py \
    --train \
    --framework UHop \
    --model HR_BiLSTM_plus \
    --dataset kinship \
    --emb_size 300 \
    --hidden_size 100 \
    --dynamic recurrent \
    --epoch_num "${epochs}" \
    --seed "${seed}" \
    --device "${device}" \
    --path "${checkpoint_dir}"
