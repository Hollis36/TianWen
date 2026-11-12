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
#
# Usage:
#   KAGGLE_USERNAME=... KAGGLE_KEY=... DATASET_SLUG=owner/name ./scripts/run_on_kaggle.sh push
#   ./scripts/run_on_kaggle.sh poll      # check status; fetch result when complete
#
# The token is written to ~/.kaggle/kaggle.json (outside the repo) and never committed.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${TIANWEN_KAGGLE_WORK:-/tmp/tianwen-kaggle}"
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
  sed -e "s/^MAX_STEPS = .*/MAX_STEPS = ${steps}/" \
      -e "s/^LIMIT_VAL_BATCHES = .*/LIMIT_VAL_BATCHES = ${limit_val}/" \
      -e "s/^DISTILL_MODE = .*/DISTILL_MODE = \"${mode}\"/" \
      "$REPO_DIR/kaggle/run.py" > "$WORK/run.py"
  sed -e "s#__USERNAME__#${KAGGLE_USERNAME}#" -e "s#__DATASET_SLUG__#${DATASET_SLUG}#" \
      "$REPO_DIR/kaggle/kernel-metadata.json" > "$WORK/kernel-metadata.json"
  echo "${KAGGLE_USERNAME}/tianwen-ablation" > "$SLUG_FILE"
  echo "Pushing kernel ${KAGGLE_USERNAME}/tianwen-ablation (dataset=${DATASET_SLUG}, steps=${steps})..."
  kaggle kernels push -p "$WORK"
  echo "Pushed. Poll with: ./scripts/run_on_kaggle.sh poll"
}

cmd_poll() {
  write_creds
  local slug
  slug="$(cat "$SLUG_FILE" 2>/dev/null || echo "${KAGGLE_USERNAME:-}/tianwen-ablation")"
  local status
  status="$(kaggle kernels status "$slug" 2>&1 | tr -d '\r')"
  echo "$status"
  if echo "$status" | grep -qiE "complete"; then
    kaggle kernels output "$slug" -p "$WORK/out" >/dev/null 2>&1 || true
    echo "=== ablation result ==="
    if [ -f "$WORK/out/ablation_result.json" ]; then
      cat "$WORK/out/ablation_result.json"
      # Persist into the repo so results survive ephemeral containers.
      mkdir -p "$REPO_DIR/results"
      local name
      name="$(python3 -c "import json;d=json.load(open('$WORK/out/ablation_result.json'));print(f\"{d.get('dataset','run')}_{d.get('detector','det')}_{d.get('distill_mode','feature')}_{d.get('max_steps','x')}steps\")")"
      cp "$WORK/out/ablation_result.json" "$REPO_DIR/results/${name}.json"
      echo "Saved to results/${name}.json (commit it to keep it)."
    else
      grep -ah "ABLATION_RESULT_JSON" "$WORK"/out/*.log 2>/dev/null || echo "(no result file; see $WORK/out)"
    fi
  elif echo "$status" | grep -qiE "error|cancel"; then
    echo "Kernel failed; fetching logs..."
    kaggle kernels output "$slug" -p "$WORK/out" >/dev/null 2>&1 || true
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
