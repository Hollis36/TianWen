#!/usr/bin/env bash
#
# Drive a TianWen distillation ablation on Kaggle GPU via the Kaggle API.
#
# Requires:
#   KAGGLE_USERNAME, KAGGLE_KEY   - Kaggle API token (Account -> Settings -> API)
#   DATASET_SLUG                  - Kaggle dataset to attach, e.g. awsaf49/coco-2017-dataset
# Optional:
#   MAX_STEPS                     - training steps per arm (default 2000)
#   DISTILL_MODE                  - dense (default) | feature | logit | response
#   ACCELERATOR                   - Kaggle accelerator (default NvidiaTeslaT4)
#   DISTILL_WEIGHT                - distillation loss weight (default 1.0)
#   TRAIN_FRACTION                - fraction of the train split to use (default 1.0)
#   VLM_MODEL                     - teacher, e.g. openai/clip-vit-large-patch14-336
#   MAX_EPOCHS                    - converged multi-arm protocol (epochs; overrides MAX_STEPS)
#   PATIENCE / DEV_SIZE / ARMS    - early stopping, dev split size, arms (baseline,clip,random)
#   SCRIPT                        - run (distillation, default) | ssl (pseudo-label arm)
#   SSL_ARM / STUDENT_EPOCHS / LABELLER_EPOCHS / IMGSZ / CONF - pseudo-label arm settings
#   TIANWEN_REF                   - TianWen branch/tag the kernel clones (default main)
#   SEED                          - seed shared by both arms (default 0)
#   KERNEL_NAME                   - kernel slug (default tianwen-ablation); use distinct
#                                   names to run several arms in parallel
#
# Usage:
#   KAGGLE_USERNAME=... KAGGLE_KEY=... DATASET_SLUG=owner/name ./scripts/run_on_kaggle.sh push
#   ./scripts/run_on_kaggle.sh poll      # check status; fetch result when complete
#
# The token is written to ~/.kaggle/kaggle.json (outside the repo) and never committed.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_NAME="${KERNEL_NAME:-tianwen-ablation}"
WORK="${TIANWEN_KAGGLE_WORK:-/tmp/tianwen-kaggle/${KERNEL_NAME}}"
SLUG_FILE="$WORK/slug.txt"

write_creds() {
  # Prefer an existing ~/.kaggle/kaggle.json (so the key never needs to be on the
  # command line); otherwise build it from KAGGLE_USERNAME / KAGGLE_KEY.
  if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    if [ -z "${KAGGLE_USERNAME:-}" ]; then
      KAGGLE_USERNAME="$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")"
      export KAGGLE_USERNAME
    fi
    return
  fi
  : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
  : "${KAGGLE_KEY:?set KAGGLE_KEY}"
  mkdir -p "$HOME/.kaggle"
  printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > "$HOME/.kaggle/kaggle.json"
  chmod 600 "$HOME/.kaggle/kaggle.json"
}

cmd_push() {
  : "${DATASET_SLUG:?set DATASET_SLUG, e.g. awsaf49/coco-2017-dataset}"
  write_creds
  local steps="${MAX_STEPS:-2000}"
  local limit_val="${LIMIT_VAL:-None}"
  local mode="${DISTILL_MODE:-dense}"
  mkdir -p "$WORK"
  local src="$REPO_DIR/kaggle/run.py"
  [ "${SCRIPT:-run}" = "ssl" ] && src="$REPO_DIR/kaggle/ssl_run.py"
  sed -e "s/^MAX_STEPS = .*/MAX_STEPS = ${steps}/" \
      -e "s/^LIMIT_VAL_BATCHES = .*/LIMIT_VAL_BATCHES = ${limit_val}/" \
      -e "s/^DISTILL_MODE = .*/DISTILL_MODE = \"${mode}\"/" \
      -e "s/^SEED = .*/SEED = ${SEED:-0}/" \
      -e "s/^MAX_EPOCHS = .*/MAX_EPOCHS = ${MAX_EPOCHS:-None}/" \
      -e "s/^PATIENCE = .*/PATIENCE = ${PATIENCE:-10}/" \
      -e "s/^DEV_SIZE = .*/DEV_SIZE = ${DEV_SIZE:-0}/" \
      -e "s/^ARMS = .*/ARMS = \"${ARMS:-baseline,clip,random}\"/" \
      -e "s#^TIANWEN_REF = .*#TIANWEN_REF = \"${TIANWEN_REF:-main}\"#" \
      -e "s/^DISTILL_WEIGHT = .*/DISTILL_WEIGHT = ${DISTILL_WEIGHT:-1.0}/" \
      -e "s/^TRAIN_FRACTION = .*/TRAIN_FRACTION = ${TRAIN_FRACTION:-1.0}/" \
      -e "s#^VLM_MODEL = .*#VLM_MODEL = \"${VLM_MODEL:-openai/clip-vit-base-patch32}\"#" \
      -e "s/^SSL_ARM = .*/SSL_ARM = \"${SSL_ARM:-supervised}\"/" \
      -e "s/^LABELED_FRACTION = .*/LABELED_FRACTION = ${TRAIN_FRACTION:-0.1}/" \
      -e "s/^STUDENT_EPOCHS = .*/STUDENT_EPOCHS = ${STUDENT_EPOCHS:-60}/" \
      -e "s/^LABELLER_EPOCHS = .*/LABELLER_EPOCHS = ${LABELLER_EPOCHS:-60}/" \
      -e "s/^IMGSZ = .*/IMGSZ = ${IMGSZ:-320}/" \
      -e "s/^CONF = .*/CONF = ${CONF:-0.5}/" \
      "$src" > "$WORK/run.py"
  sed -e "s#__USERNAME__/tianwen-ablation#${KAGGLE_USERNAME}/${KERNEL_NAME}#" \
      -e "s#\"title\": \"tianwen-ablation\"#\"title\": \"${KERNEL_NAME}\"#" \
      -e "s#__DATASET_SLUG__#${DATASET_SLUG}#" \
      "$REPO_DIR/kaggle/kernel-metadata.json" > "$WORK/kernel-metadata.json"
  echo "${KAGGLE_USERNAME}/${KERNEL_NAME}" > "$SLUG_FILE"
  echo "Pushing kernel ${KAGGLE_USERNAME}/${KERNEL_NAME} (dataset=${DATASET_SLUG}, steps=${steps})..."
  kaggle kernels push -p "$WORK" --accelerator "${ACCELERATOR:-NvidiaTeslaT4}"
  echo "Pushed. Poll with: ./scripts/run_on_kaggle.sh poll"
}

cmd_poll() {
  write_creds
  local slug
  slug="$(cat "$SLUG_FILE" 2>/dev/null || echo "${KAGGLE_USERNAME:-}/${KERNEL_NAME}")"
  local status
  status="$(kaggle kernels status "$slug" 2>&1 | tr -d '\r')"
  echo "$status"
  if echo "$status" | grep -qiE "complete"; then
    # Only the result JSON and the run log: the SSL kernels leave thousands of
    # split images in /kaggle/working, which otherwise take ~1 h to download.
    kaggle kernels output "$slug" -p "$WORK/out" \
      --file-pattern '(ablation_result\.json|\.log)$' >/dev/null 2>&1 || true
    echo "=== ablation result ==="
    if [ -f "$WORK/out/ablation_result.json" ]; then
      cat "$WORK/out/ablation_result.json"
      # Persist into the repo so results survive ephemeral containers.
      mkdir -p "$REPO_DIR/results"
      local name
      name="$(python3 "$REPO_DIR/scripts/result_name.py" "$WORK/out/ablation_result.json")"
      cp "$WORK/out/ablation_result.json" "$REPO_DIR/results/${name}.json"
      echo "Saved to results/${name}.json (commit it to keep it)."
    else
      grep -ah "ABLATION_RESULT_JSON" "$WORK"/out/*.log 2>/dev/null || echo "(no result file; see $WORK/out)"
    fi
  elif echo "$status" | grep -qiE "error|cancel"; then
    echo "Kernel failed; fetching logs..."
    kaggle kernels output "$slug" -p "$WORK/out" \
      --file-pattern '(ablation_result\.json|\.log)$' >/dev/null 2>&1 || true
    ls -1 "$WORK/out" 2>/dev/null || true
  else
    echo "(still running — poll again later)"
  fi
}

case "${1:-}" in
  push) cmd_push ;;
  poll) cmd_poll ;;
  *) echo "usage: $0 {push|poll}" >&2; exit 2 ;;
esac
