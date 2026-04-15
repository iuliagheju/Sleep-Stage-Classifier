# Sleep Stager

A test-first, spec-driven pipeline for classifying 30 s EEG epochs into W/N1/N2/N3/REM with calibrated probabilities and HMM-based temporal smoothing. The repository operationalizes the requirements documented in `requirements.txt` and `docs/evaluation_spec.md`.

## Highlights
- Subject-wise data splits, deterministic seeds, and artifact logging for every run.
- Multiple model families (classical ML, 1D CNN, CNN-BiLSTM) behind a shared training interface.
- Calibration tooling (temperature scaling, reliability diagrams, Brier/ECE metrics) and HMM post-processing guardrails.
- Smoke fixtures and CLI smoke tests to keep the pipeline reproducible on commodity laptops.

## Quick Start
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -e .[dev]
pre-commit install
```

Fetch or point to the PhysioNet Sleep-EDF 1.0.0 release:
```bash
scripts/fetch_sleep_edf.sh
setx DATA_DIR "C:/path/to/data/raw/sleep-edfx-1.0.0"
```
Requires `wfdbdownload` (WFDB toolkit) and PhysioNet credentials when prompted.

Run the default experiment (configurable via Hydra):
```bash
python -m sleep_stager.cli.train --config-path configs --config-name default
```

Or use the installed console script:
```bash
sleep-stager --config-path configs --config-name default
```

Spec CLI alias:
```bash
sleepstage train --config-path configs --config-name default
```

## Model + Feature Overrides
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

## Evaluation Protocols
Run a single k-fold split (subject-wise CV):
```bash
python -m sleep_stager.cli.train --config-path configs --config-name default \
  --override evaluation.protocol=kfold_subject \
  --override evaluation.fold_index=0
```

## Additional CLI Commands
```bash
sleepstage make-fixture --out data/fixtures/smoke --num-subjects 3 --epochs-per-subject 20
sleepstage make-fixture --out data/fixtures/edf --raw-dir data/raw/sleep-edfx-1.0.0 --num-pairs 10
sleepstage fetch-data --dest data/raw/sleep-edfx-1.0.0
sleepstage train-kfold --config-path configs --config-name default --runs-root artifacts/kfold
sleepstage eval --run artifacts/<run_id>
sleepstage eval-folds --runs-root artifacts
```

## Test-First Loop
1. Update specs/contracts.
2. Add failing tests (unit or smoke).
3. Implement the minimal feature.
4. Run `pytest -q` (and `pytest -q -m smoke` when touching the CLI).
5. Commit with scope + tests run.

## Additional Docs
- `docs/evaluation_spec.md`: locked evaluation rules (subjects, metrics, guardrails).
- `configs/default.yaml`: training + evaluation defaults, including HMM tolerance gates.

Artifacts for each run (metrics, per-subject tables, confusion matrix, calibration figures, predictions, configs, checkpoints) are written under `artifacts/<timestamp>` with the git commit hash embedded in `metrics.json`.
