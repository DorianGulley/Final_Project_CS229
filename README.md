# Clash Royale S18 Training & Deck Analysis Pipeline

This repository contains a complete ML pipeline for predicting Clash Royale battle outcomes
and a small suite of analysis / adversarial-deck tools that use the trained models to:

- Build sparse features from raw Kaggle battle logs
- Train and evaluate multiple models (Logistic Regression, AdaBoost, XGBoost, MLP)
- Save/load precomputed feature artifacts (fast iteration and cloud training)
- Analyze counters and generate adversarial decks (scripts under the repo root)

This README has concise, accurate usage notes for the core training flow, cloud runs,
and the analysis tools (`explore_decks.py`, `oracle_eval.py`, `counter_analysis.py`, `greedy_search.py`).

## Quick Start — Local training

Run the default training (Logistic Regression, deck+pairwise+delta features):

```bash
python train.py
```

Quick iteration mode (loads a single day's CSVs):

```bash
python train.py --quick
```

Train a specific model:

```bash
python train.py --model xgb
python train.py --model adaboost
python train.py --model nn --hidden-layers 256 128 64
```

Notes on `--hidden-layers` for `--model nn`:
- On the command line provide space-separated integers, e.g. `--hidden-layers 256 128 64`.
- When submitting a Vertex job with `gcloud --args`, tokens are comma-separated, e.g.
  `--args=--model,nn,--hidden-layers,256,128,64,...` (this is a `gcloud`/shell detail).

Important training flags (examples):

- `--features-path`: local or `gs://` path to load precomputed artifacts (skip feature build)
- `--save-features`: save built artifacts to local or `gs://` for reuse
- `--use-local <path>`: treat a directory (or `gs://...`) as the dataset root
- `--gcs-direct`: when combined with `--use-local gs://...`, read CSVs directly from GCS
- `--quick`: load only the Dec 27, 2020 quick CSV (~1GB) for fast iteration
- `--n-jobs`: number of threads/cores to pass to LogReg/XGBoost (default `1`)
- `--plot-loss <path>`: save training loss curve locally or to `gs://`

## Feature Persistence (two-phase workflow)

1) Precompute and save feature artifacts once (Phase 1):

```bash
python train.py --quick --save-features gs://my-bucket/features/v1
```

2) Reuse artifacts for fast experiments (Phase 2):

```bash
python train.py --features-path gs://my-bucket/features/v1 --model logreg
python train.py --features-path /tmp/features/v1 --model xgb
```

Artifacts saved: `X_train.npz`, `X_val.npz`, `X_test.npz`, `y_*.npy`, and `metadata.json`.

## Adversarial / Analysis Scripts

This repository includes several convenience scripts that leverage the trained models
to analyze decks and search for counter / adversarial decks:

- `explore_decks.py` — quick frequency and head-to-head analysis of decks (CLI: `--quick`, `--sample`)
- `oracle_eval.py` — evaluates a trained model as a matchup oracle (compares predicted vs empirical WRs)
- `counter_analysis.py` — extracts per-card counter-scores from model coefficients and finds real decks
- `greedy_search.py` — hill-climb / greedy local search to generate adversarial counter-decks against a target

Usage examples:

```bash
# Basic exploration (fast)
python explore_decks.py --quick

# Evaluate model as oracle
python oracle_eval.py --quick --min-games 20 --top-k 10

# Compute top counter-cards and find real decks that contain them
python counter_analysis.py --quick --top-cards 20

# Greedy local search for an adversarial deck against rank-4 target
python greedy_search.py --quick --target-rank 4 --lambda 0.6 --elixir-cap 4.0
```

Each script's `--help` describes the available options; they all rely on the same
feature-building and model adapter code so outputs are compatible with `train.py` artifacts.

## Cloud / Vertex AI

Recommended workflow for cloud training with Vertex AI:

1. Precompute features and upload to GCS (`--save-features gs://...`).
2. Submit separate Vertex custom jobs (one per model) that load `--features-path` from GCS.

Example `gcloud` job (GPU) that reads CSVs directly from GCS and runs the NN in quick mode:

```bash
gcloud ai custom-jobs create \
  --region=us-west2 \
  --display-name="clash-royale-nn-quick-gpu-gcs" \
  --config=- <<'EOF'
workerPoolSpecs:
  - machineSpec:
      machineType: "n1-standard-8"
      acceleratorConfig:
        type: "NVIDIA_TESLA_T4"
        count: 1
    replicaCount: 1
    containerSpec:
      imageUri: "gcr.io/my-cr-project-data/clash-royale:latest"
      args:
        - "--model"
        - "nn"
        - "--quick"
        - "--use-local"
        - "gs://my-cr-project-data/20"
        - "--gcs-direct"
        - "--n-jobs"
        - "1"
        - "--hidden-layers"
        - "256"
        - "128"
        - "64"
        - "--plot-loss"
        - "gs://my-cr-project-data/nn_loss_curve.png"
        - "--save"
        - "gs://my-cr-project-data/metrics/nn_quick_gpu.json"
EOF
```

Notes:
- Replace `gcr.io/my-cr-project-data/...` and `gs://my-cr-project-data/...` with your image and bucket.
- Use `NVIDIA_TESLA_V100` or `A100` for higher throughput if you have quota / budget.
- Ensure the Vertex service account has access to the GCS bucket (objectViewer/objectCreator as needed).

## Implementation notes (concise)

- Feature blocks: `deck` (card presence), `ab` (pairwise anti-symmetric interactions), `delta` (trophy diff), `levels` (card-level diff).
- Sparse matrices use `scipy.sparse.csr_matrix` and are saved/loaded with `save_npz`/`load_npz`.
- Models live under `models/` as lightweight adapters with a common `fit/predict_proba/predict` API.

## Troubleshooting

- If `--features-path` cannot be found, run `python train.py --quick --save-features <path>` first.
- For GCS access issues, set `GOOGLE_APPLICATION_CREDENTIALS` or run `gcloud auth application-default login`.
- If feature building OOMs: use `--quick`, `--sample`, or precompute features on a larger machine.

## Repo layout

```
. 
├── train.py
├── data.py
├── features.py
├── feature_io.py
├── metrics.py
├── models/
│   ├── __init__.py
│   ├── logreg.py
│   ├── adaboost.py
│   ├── xgb.py
│   └── nn.py
├── greedy_search.py
├── explore_decks.py
├── oracle_eval.py
├── counter_analysis.py
├── cloud_utils.py
├── Dockerfile
└── requirements.txt
```

---

**Last Updated**: December 2025
