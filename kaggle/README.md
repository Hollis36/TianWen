# Run the ablation on Kaggle GPU (remote-driven)

Drive a TianWen distillation ablation on a free Kaggle GPU via the Kaggle API —
no local GPU needed. It runs the controlled experiment (VLM teacher off vs on)
and returns baseline / distilled / Δ mAP.

## One-time
1. Kaggle account → **Settings → API → Create New Token** (downloads
   `kaggle.json` with your `username` + `key`). Phone-verify the account
   (required for GPU + internet kernels).
2. Add the dataset on Kaggle and note its slug, e.g. `awsaf49/coco-2017-dataset`
   (COCO) or a NEU-DET / GC10 YOLO-format dataset.

## Run
```bash
export KAGGLE_USERNAME=...  KAGGLE_KEY=...
export DATASET_SLUG=owner/dataset-name      # COCO or a YOLO-format defect dataset
export MAX_STEPS=2000                        # optional, steps per arm

./scripts/run_on_kaggle.sh push              # pushes + starts the GPU kernel
./scripts/run_on_kaggle.sh poll              # repeat until it prints the result
```

`run.py` (the kernel) clones TianWen, auto-detects the attached dataset (COCO via
`discover_coco`, else the first YOLO `data.yaml`), runs the ablation, and writes
`ablation_result.json`. The token is stored only in `~/.kaggle/kaggle.json`
(outside the repo) and can be revoked on Kaggle at any time.
