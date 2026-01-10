# Evaluation Specification (Locked)

**Dataset:** PhysioNet Sleep-EDF Expanded (sleep-edfx/1.0.0). Only EEG Fpz-Cz + EOG horizontal are considered for v1. EDF files stay in `data/raw/sleep-edfx-1.0.0`. The pipeline consumes pre-extracted 30 s epochs with labels \{W, N1, N2, N3, REM\}.

**Splitting protocol:** subject-wise only. Each subject appears in exactly one of {train, val, test}. Random seeds are pinned to `42` (numpy) and `1337` (torch). Splits are recorded in `artifacts/<run_id>/splits.json` and validated via automated tests to avoid leakage.

**Metrics:**
- Macro F1 (primary), per-class F1, overall accuracy, balanced accuracy.
- Calibration: Expected Calibration Error (ECE, 15 equal-width bins) and multiclass Brier score.
- Temporal consistency: implausible transition rate (custom rule set) and stage-change rate before/after HMM smoothing.
- Efficiency: unit tests <= 60 seconds, end-to-end smoke <= 10 minutes, baseline train <= 2 minutes, one-epoch deep model <= 8 minutes, postprocess <= 2 minutes, inference throughput >= 200 epochs/s on CPU, parameter caps per specs/model_limits.yaml (cnn 1M, seq 3M, attention 5M).

**Acceptance gates (v1):**
1. Reproducibility: rerunning the same config twice produces macro F1 difference < 0.005.
2. No data leakage: subject overlap tests must pass.
3. Calibration metrics (ECE, Brier) computed and saved for every evaluation.
4. HMM smoothing macro F1 drop <= 0.01 on the smoke fixture.

**Artifacts per run:** metrics.json (with git SHA + config copy), per-subject CSV, confusion matrix (PNG + JSON), calibration plot, reliability bins, predictions (parquet), checkpoint, logs, transition diagnostics.

**Commands:**
- `pytest -q` (unit tests) on each commit.
- `pytest -q -m smoke` before merging.
- `python -m sleep_stager.cli.train --config-path configs --config-name default` for default experiment.
- `sleepstage train-kfold --config-path configs --config-name default --runs-root artifacts/kfold` for secondary protocol runs.
- `sleepstage eval-folds --runs-root artifacts` to aggregate k-fold runs.
