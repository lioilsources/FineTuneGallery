#!/usr/bin/env bash
# Train one built dataset package on the GPU box and deploy the LoRA to ComfyUI.
#
#   ./run.sh <dataset-name>
#
# Configuration via train/.env (see .env.example): NAS_DATASETS, CKPT_DIR,
# LORA_DIR. The dataset package is produced by the finetune-gallery backend
# (Datasets → Build) and is fully self-contained: img/ + captions + both tomls.
set -euo pipefail

cd "$(dirname "$0")"
[ -f .env ] && source .env

DATASET="${1:?usage: run.sh <dataset-name>}"
: "${NAS_DATASETS:?set NAS_DATASETS in train/.env (local mount path or user@nas:/path)}"
: "${CKPT_DIR:?set CKPT_DIR in train/.env (ComfyUI checkpoints dir)}"
: "${LORA_DIR:?set LORA_DIR in train/.env (ComfyUI loras dir)}"
IMAGE="${IMAGE:-finetune-kohya:latest}"

WORK="$(pwd)/work/$DATASET"
mkdir -p "$WORK"

echo "── sync dataset $DATASET"
case "$NAS_DATASETS" in
  *:*) rsync -a --delete "$NAS_DATASETS/$DATASET/" "$WORK/" ;;   # remote (ssh)
  *)   rsync -a --delete "$NAS_DATASETS/$DATASET/" "$WORK/" ;;   # local mount
esac
[ -f "$WORK/train_config.toml" ] || { echo "train_config.toml missing — did you Build the dataset?"; exit 1; }

# Resolve the checkpoint sentinel for the in-container mount layout.
sed -i.bak 's#__CKPT_DIR__#/ckpts#' "$WORK/train_config.toml" && rm -f "$WORK/train_config.toml.bak"
OUTPUT_NAME="$(sed -n 's/^output_name = "\(.*\)"/\1/p' "$WORK/train_config.toml")"
mkdir -p "$WORK/output"

echo "── train ($OUTPUT_NAME)"
docker run --rm --gpus all \
  -v "$WORK":/dataset \
  -v "$CKPT_DIR":/ckpts:ro \
  -v finetune-kohya-cache:/cache \
  "$IMAGE" \
  sdxl_train_network.py \
    --config_file /dataset/train_config.toml \
    --dataset_config /dataset/dataset.toml

FINAL="$WORK/output/$OUTPUT_NAME.safetensors"
[ -f "$FINAL" ] || { echo "expected output $FINAL not found"; exit 1; }

# Guard the app's LoRA-family heuristic: SDXL LoRA filenames must not
# contain "flux" or the Ol1nLLM picker files them under the Flux family.
case "$(basename "$FINAL" | tr '[:upper:]' '[:lower:]')" in
  *flux*) echo "refusing to deploy: filename contains 'flux'"; exit 1 ;;
esac

echo "── deploy → $LORA_DIR/$OUTPUT_NAME.safetensors"
cp "$FINAL" "$LORA_DIR/$OUTPUT_NAME.safetensors"
echo "done — the app picks it up automatically via ComfyUI /object_info/LoraLoader."
echo "Epoch checkpoints kept in $WORK/output for strength/epoch comparisons."
