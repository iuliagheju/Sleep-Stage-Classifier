# Sleep Stager

A test-first, spec-driven pipeline for classifying 30 s EEG epochs into **W / N1 / N2 / N3 / REM** with calibrated probabilities and HMM-based temporal smoothing. The repository operationalizes the requirements documented in `requirements.txt` and `docs/evaluation_spec.md`.

## Project Description

Manual sleep scoring is slow and varies between specialists. This project automates the task: given raw single-channel EEG from polysomnography (PSG) recordings, it predicts the five standard sleep stages for every 30-second epoch. The pipeline is designed to be **clinically interpretable** rather than just accurate — it outputs calibrated per-epoch probabilities, applies Hidden Markov Model (HMM) smoothing to remove implausible stage transitions, and reports metrics suited to imbalanced classes (macro-F1, per-class F1, confusion matrices, calibration error).

It compares four model families behind a single training/evaluation interface:

1. **Classical ML baseline** — hand-crafted time-domain and frequency-band features fed to Random Forest / SVM / Logistic Regression.
2. **1D CNN** — learns features directly from raw EEG epochs.
3. **CNN-BiLSTM** (DeepSleepNet-style) — adds short-range temporal context across neighbouring epochs.
4. **Attention sequence model** (SeqSleepNet-style) — weights neighbouring epochs by relevance, with inspectable attention weights.

Everything is built to run reproducibly on **CPU-only commodity hardware** (no GPU required).

## Highlights
- Subject-wise data splits, deterministic seeds, and artifact logging for every run.
- Four model families (classical ML, 1D CNN, CNN-BiLSTM, attention sequence model) behind a shared training interface.
- Calibration tooling (temperature scaling, reliability diagrams, Brier/ECE metrics) and HMM post-processing guardrails.
- Spec-driven evaluation with tolerance "gates" that fail a run if it regresses below locked thresholds.
- Smoke fixtures and CLI smoke tests to keep the pipeline reproducible on commodity laptops.

## Architecture

The application is a Python package (`src/sleep_stager`) organized as a linear pipeline: **ingest → feature/transform → model → calibration → post-process → evaluate → report**. A Typer CLI orchestrates the stages, and Hydra/OmegaConf drives configuration. Behaviour is locked by declarative YAML specs under `specs/`.

```
src/sleep_stager/
├── cli/            # Typer entrypoints (main.py = unified CLI, train.py = training loop)
├── data/           # EDF loading, epoching/ingest, subject-wise splits, fixtures, raw validation
├── features/       # Hand-crafted features: time_domain, bandpower, spectral entropy, transforms
├── models/         # classical, cnn (1D CNN), seq (CNN-BiLSTM), attention, calibration, inference
├── postprocess/    # hmm.py — HMM temporal smoothing of per-epoch probabilities
├── eval/           # metrics, calibration, temporal, gates, baseline, report, schema validation
└── utils/          # logging, progress bars, RNG seeding, system/hardware info

configs/            # Hydra config (default.yaml): training + evaluation defaults, HMM gates
specs/              # Locked contracts: datasets, evaluation rules, gates, HMM transitions, schemas
docs/               # evaluation_spec.md — the authoritative evaluation rules
scripts/            # Dataset fetch + helper utilities (download, plotting, fixtures)
tests/              # Unit, integration/smoke, and golden tests mirroring the package layout
```

**Data flow**

1. `data/` loads EDF PSG/Hypnogram pairs (via MNE), slices them into 30 s epochs, and produces subject-wise train/test splits so no subject leaks across the split.
2. `features/` (classical path) or the raw epoch tensor (deep path) feeds the chosen model in `models/`.
3. `models/calibration.py` calibrates the output probabilities; `postprocess/hmm.py` smooths the predicted sequence using transition probabilities estimated from training data.
4. `eval/` computes metrics, calibration diagnostics, and temporal consistency, then `eval/gates.py` checks them against locked thresholds.
5. Each run writes a self-describing artifact directory (`metrics.json`, per-subject tables, confusion matrix, calibration figures, predictions, config, checkpoint) with the git commit hash embedded for provenance.

## Technologies Used

| Area | Libraries |
|------|-----------|
| Core numerics | NumPy, SciPy, pandas, pyarrow |
| Classical ML | scikit-learn |
| Deep learning | PyTorch (torch, torchvision, torchaudio) |
| EEG / EDF I/O | MNE, h5py |
| CLI & config | Typer, Hydra-core, OmegaConf, Pydantic, PyYAML |
| Visualization & UX | matplotlib, seaborn, Rich |
| Tooling (dev) | pytest, pytest-cov, pytest-mock, black, isort, mypy, pre-commit |

Language: **Python ≥ 3.10**. 

## Requirements & Installation

Create a virtual environment and install the package in editable mode with dev extras:

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -e .[dev]
pre-commit install
```

This installs all runtime dependencies declared in `pyproject.toml` (mirrored in `requirements.txt`) plus the development/test toolchain. The `pre-commit` hooks (black, isort, mypy) run automatically on commit.

## Downloading the Dataset

> **Note:** the dataset is **not** included in this repository — `data/` is gitignored. You must download it yourself before training.

The project uses the PhysioNet **Sleep-EDF Expanded (Sleep-EDFx) 1.0.0** dataset (≈ 8.1 GB). Get it from PhysioNet using any of these official methods, into `data/raw`:

```bash
# Option A — download with wget (recommended)
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/

# Option B — download with AWS CLI (no credentials needed)
aws s3 sync --no-sign-request s3://physionet-open/sleep-edfx/1.0.0/ data/raw

# Option C — download the ZIP (8.1 GB) from the dataset page and extract it
#   https://physionet.org/content/sleep-edfx/1.0.0/
```

`wget -r` mirrors the site path, so the files land under `physionet.org/files/sleep-edfx/1.0.0/`; move that content (the `sleep-cassette/` and `sleep-telemetry/` folders) into `data/raw`. After downloading, you should end up with a directory containing the `sleep-cassette/` and/or `sleep-telemetry/` folders of `*-PSG.edf` / `*-Hypnogram.edf` files. A typical layout:

```
data/raw/
├── sleep-cassette/      # SC* PSG + Hypnogram EDF files
├── sleep-telemetry/     # ST* PSG + Hypnogram EDF files
├── RECORDS
├── SC-subjects.xls
└── ST-subjects.xls
```

The pipeline finds the EDF files via the `DATA_DIR` environment variable, which sets `data.dir` in the config (`data.dir: ${oc.env:DATA_DIR, data/fixtures}` — if `DATA_DIR` is unset it falls back to `data/fixtures`).

> **Important:** for **training**, point `DATA_DIR` at one **subset folder** — `data/raw/sleep-cassette` **or** `data/raw/sleep-telemetry` — **not** at `data/raw` itself. Each subset has its own dataset spec, so you train on one subset at a time. The training command derives the dataset from the path: it must contain `sleep-cassette`, `sleep-telemetry`, or `fixture`, otherwise it fails with `Unsupported dataset tag 'unknown'`.

```powershell
# Train on the Sleep Cassette subset (use sleep-telemetry for the ST subset)
# Persist for new terminals (Windows)
setx DATA_DIR "C:/Users/user/Documents/GitHub/Sleep-Stage-Classifier/data/raw/sleep-cassette"
# Or for the current PowerShell session only
$env:DATA_DIR = "$PWD/data/raw/sleep-cassette"
```

To **validate** the download (not train), the CLI accepts the parent folder and auto-detects the subsets underneath (see `src/sleep_stager/data/raw.py`):

```bash
# Validate an existing copy
sleepstage fetch-data --dest data/raw

# Attempt the download via scripts/fetch_sleep_edf.sh, then validate
sleepstage fetch-data --dest data/raw --download
```

If you don't have the dataset yet but want to exercise the pipeline, generate a tiny synthetic fixture instead (see User Manual → Fixtures).

## User Manual

### Run the default experiment

With `DATA_DIR` pointed at a subset folder (see Downloading the Dataset), run a training. The `artifacts.root` output path **must contain the same dataset token** as `DATA_DIR` (`sleep-cassette`, `sleep-telemetry`, or `fixtures`); otherwise the run aborts with a "dataset subset mismatch" error.

```bash
# Example: train on the Sleep Cassette subset
sleepstage train --config-path configs --config-name default \
  --override artifacts.root=artifacts/sleep-cassette/holdout
```

The default config (`configs/default.yaml`) targets the synthetic fixtures (`artifacts.root: artifacts/fixtures/holdout`), so the bare command below works only when `DATA_DIR` points at a `fixtures` path — for real data, override `artifacts.root` as shown above:

```bash
python -m sleep_stager.cli.train --config-path configs --config-name default
```

Equivalent entrypoints — the installed console script and the spec CLI alias:
```bash
sleep-stager --config-path configs --config-name default
sleepstage   train --config-path configs --config-name default
```

### Model + feature overrides
Select the classical baseline type:
```bash
python -m sleep_stager.cli.train --config-path configs --config-name default \
  --override model.classical.model_type=svm
```

Toggle feature families:
```bash
python -m sleep_stager.cli.train --config-path configs --config-name default \
  --override features.use_time_domain=false \
  --override features.use_bandpower_ratios=true \
  --override features.use_spectral_entropy=true
```

### Evaluation protocols
Run a single k-fold split (subject-wise CV):
```bash
python -m sleep_stager.cli.train --config-path configs --config-name default \
  --override evaluation.protocol=kfold_subject \
  --override evaluation.fold_index=0
```

Run full cross-validation:
```bash
sleepstage train-kfold --config-path configs --config-name default --runs-root artifacts/kfold
```

### Fixtures (no dataset required)
```bash
# Synthetic smoke fixture
sleepstage make-fixture --out data/fixtures/smoke --num-subjects 3 --epochs-per-subject 20
# Subset built from real EDF pairs
sleepstage make-fixture --out data/fixtures/edf --raw-dir data/raw --num-pairs 10
```

### Evaluation & reporting
```bash
sleepstage eval           --run artifacts/<run_id>           # validate + summarize one run
sleepstage eval-folds     --runs-root artifacts/kfold        # aggregate k-fold runs
sleepstage list-runs      --runs-root artifacts              # tabular overview of runs
sleepstage summarize-runs --runs-root artifacts              # schema-validated dataset summary
sleepstage compare-runs   --runs-root artifacts              # cross-model comparison report
```

Artifacts for each run (metrics, per-subject tables, confusion matrix, calibration figures, predictions, configs, checkpoints) are written under `artifacts/<timestamp>` with the git commit hash embedded in `metrics.json`.

## Testing

The project follows a **test-first** workflow, and the `tests/` tree mirrors the package layout. Test types:

- **Unit tests** — per module: `tests/features/`, `tests/models/`, `tests/eval/`, `tests/data/`, `tests/postprocess/`, `tests/utils/`.
- **Determinism / invariants** — e.g. `test_bandpower_deterministic.py`, `test_probabilities_simplex.py`, `test_dataset_invariants.py`, `test_context_window.py` enforce reproducible features and valid probability simplexes.
- **Schema & provenance** — `test_schema.py`, `test_summary_schema.py`, `test_provenance.py` validate that artifacts match the locked YAML specs and carry git/commit metadata.
- **Gates** — `test_gates.py` checks the tolerance thresholds that fail a regressing run.
- **Integration / smoke** — `tests/integration/test_cli_smoke.py` and `tests/cli/*` exercise the full CLI end-to-end (marked `smoke`).
- **Golden test** — `tests/golden/metrics_smoke.json` pins expected smoke metrics.

Run them:
```bash
pytest -q                 # full unit suite
pytest -q -m smoke        # slow CLI smoke tests
pytest -q --cov           # with coverage (pytest-cov; branch coverage configured)
```

### Test-first loop
1. Update specs/contracts.
2. Add failing tests (unit or smoke).
3. Implement the minimal feature.
4. Run `pytest -q` (and `pytest -q -m smoke` when touching the CLI).
5. Commit with scope + tests run.

## Additional Docs
- `docs/evaluation_spec.md`: locked evaluation rules (subjects, metrics, guardrails).
- `configs/default.yaml`: training + evaluation defaults, including HMM tolerance gates.
- `specs/`: declarative contracts for datasets, evaluation, gates, HMM transitions, and artifact schemas.
