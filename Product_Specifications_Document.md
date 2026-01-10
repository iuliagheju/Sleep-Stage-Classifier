# sleep staging from eeg: prd builder + ai coding agent prompt (test-first, git-first)

## role
You are an AI coding agent building a reproducible, clinically interpretable sleep-staging pipeline that classifies 30 s EEG epochs into W, N1, N2, N3, REM with strong subject-wise generalization and well-calibrated probabilities. You will implement multiple model families (classical ML, 1D CNN, context-aware CNN-BiLSTM, attention-based seq model) and an HMM post-processor to improve temporal consistency.

You must work test-driven and spec-driven:
1) define success criteria and tests before feature implementations
2) implement in small increments
3) run unit + integration tests after each increment
4) commit after each significant increment with clear messages
5) continuously re-run regression tests to ensure old behavior still holds

Leave "agent notes" sections blank for future self-comments.

---

## 1) product goals (what "done" means)

### primary outcomes
A. repeatable pipeline: one command trains a chosen model, evaluates subject-wise generalization, outputs metrics + calibrated probabilities + artifacts.  
B. clinically plausible outputs: stage sequences avoid physiologically implausible jumps more often after HMM smoothing (quantified).  
C. calibrated probabilities: probabilities correspond to empirical correctness rates (quantified).  
D. efficient: training/inference is computationally reasonable on commodity hardware (quantified by run-time budgets and model size caps you set explicitly).

### non-goals (explicitly out of scope unless later added)
- multi-channel montage optimization beyond what dataset provides
- apnea/arousal/event detection
- clinical diagnosis claims

### agent notes (write here)




---

## 2) success criteria (must be measurable, defined before coding)

Define the exact dataset(s) and split protocol in a short "evaluation spec" file early, then lock it (version control it). Subject-wise generalization is mandatory.

### required evaluation protocol
- pinned dataset: Sleep-EDFx v1.0.0 sleep-telemetry (ST*) only
- headline protocol: LOSO by subject; both nights for a subject stay in the same fold
- secondary protocol (optional): 5-fold subject-wise CV; both nights stay together
- subject-wise split only (no epoch leakage across subjects)
- report per-subject metrics and macro-averages across subjects
- calibration split: 80/20 by subjects within each training fold; calibration never uses test subjects
- fixed random seeds and deterministic settings where feasible

### required metrics
Classification:
- macro F1 across stages (primary)
- per-class F1, confusion matrix
- balanced accuracy and overall accuracy (secondary)

Probability calibration:
- expected calibration error (ECE) with 15 equal-width bins
- Brier score (multi-class)
- reliability diagrams saved as artifacts (image files)
- temperature scaling is mandatory for deep models (fit on calibration split only)

Temporal consistency:
- percentage of "implausible transitions" per established transition rules you encode (rules must be documented and versioned)
- average number of stage changes per hour (or per N epochs) before vs after HMM smoothing (descriptive only)

Efficiency:
- CPU-only smoke budgets on fixture subset:
  - unit tests <= 60 seconds total
  - end-to-end smoke run <= 10 minutes
  - baseline (classical ML) training <= 2 minutes within smoke
  - one-epoch deep-model training <= 8 minutes within smoke
  - calibration + HMM + metrics + plots <= 2 minutes within smoke
- inference throughput on CPU >= 200 epochs/second on a single core (measure with fixed batch size and 30 s epochs)
- model parameter caps (report param count for every model):
  - 1D CNN <= 1 million parameters
  - CNN-BiLSTM <= 3 million parameters
  - attention seq model <= 5 million parameters

### acceptance gates (must all pass for "v1")
- reproducibility: deterministic CPU smoke run yields identical predictions + metrics
- full-run reproducibility: macro F1 within +/-0.005 and ECE within +/-0.005 across two runs (same machine, same commit, same seeds)
- if GPU nondeterminism prevents this, allow +/-0.01 and document the cause and final tolerance
- no data leakage: automated test proves split integrity
- calibration metrics computed and saved for every run
- HMM smoothing cannot reduce macro F1 by more than 0.01 on the smoke set
- HMM smoothing must reduce implausible transitions by at least 20% (relative) on the smoke set

### v1 performance gates (headline LOSO on full ST dataset; tighten after baseline)
- classical ML baseline macro F1 >= 0.65
- 1D CNN macro F1 >= (baseline + 0.03) and >= 0.68
- CNN-BiLSTM macro F1 >= (baseline + 0.05) and >= 0.70
- attention seq model macro F1 >= (baseline + 0.05) and >= 0.70
- ECE after calibration <= 0.05; ECE before calibration <= 0.10
- Brier score after calibration <= 0.55; NLL is always reported

### agent notes (write here)




---

## 3) data provisioning (agent must implement first)

### canonical source
- PhysioNet sleep-edfx/1.0.0 is the canonical source of raw data (sleep-telemetry/ST* subset only).

### hard requirements
- The repo must include `scripts/fetch_sleep_edf.sh` that downloads the ST* PSG.edf + Hypnogram.edf pairs into `data/raw/sleep-edfx-1.0.0/sleep-telemetry/` (or a consistent folder name) and verifies integrity using `SHA256SUMS.txt` when available.
- `data/` must be in `.gitignore`. No raw EDF data is committed.
- The pipeline must accept `DATA_DIR` as a config/env var and must fail with a clear error if data is missing.
- Tests must not depend on the full dataset. Implement a tiny "fixture subset" builder (e.g., `scripts/make_fixture_subset.py`) that selects N PSG/Hypnogram pairs (default N=10 pairs => 20 files), keeps pair integrity, and writes a manifest (subject_id, night, file paths) into `data/fixtures/` for smoke integration tests.
- Fixture data is for smoke/dev only and must not be used for headline metrics.

### agent notes (write here)




---

## 4) test strategy (build this first)

### testing layers
Unit tests (fast, no GPU required):
- data validation: epoch shape, sampling rate handling, label mapping, missing values
- splitting: subject-wise split integrity (no subject ID overlap between train/val/test)
- feature extraction: deterministic outputs for synthetic signals (sinusoids, white noise) and known bandpower properties
- model I/O contracts: forward pass shape, probability simplex (rows sum to 1, non-negative), batch handling
- metrics: confusion matrix correctness, macro F1 computation matches a trusted reference
- calibration: ECE and Brier computations correct on toy examples with known answers
- HMM: transition matrix normalization, Viterbi decode shape, smoothing preserves label set

Integration tests (slower, but small):
- end-to-end smoke run on fixture data: ingest -> preprocess -> split -> train (1-2 epochs) -> predict -> calibrate -> HMM -> metrics -> artifact export
- CLI contract test: commands return 0, write expected files, and log key fields

Regression tests:
- golden-file approach for tiny runs: store expected metrics JSON and compare within tolerance
- backward compatibility: past configs still run

CI:
- run unit tests on every commit
- run smoke integration tests on PR/merge
- fail fast on lint/type errors if you adopt them

### definition of "test-first" for this repo
- no new feature code without at least one failing test that proves the feature is missing or broken
- bug fixes must include a regression test that failed before the fix

### agent notes (write here)




---

## 5) repository design (spec-driven contracts)

### recommended language and stack
- Python:
  - numpy/scipy for signal operations
  - scikit-learn for classical baselines
  - PyTorch for deep models
  - HMM via a small in-repo Viterbi implementation (preferred) or a pinned, audited library

### hard requirements
- configuration-driven runs (yaml or toml) with fully logged configs copied into run artifacts
- strict separation:
  - data (ingest, preprocessing, splits)
  - features (engineered features for ML)
  - models (ML, CNN, CNN-RNN, seq-attn, calibration)
  - postprocess (HMM)
  - eval (metrics, calibration plots)
  - cli (entry points)
- every module has documented input/output contracts (types, shapes, units)

### artifact outputs per run
- `metrics.json` (all metrics)
- per-subject metrics table (csv)
- confusion matrix (png + raw counts json)
- calibration artifacts (reliability diagram png + ece/brier in json)
- predicted probabilities and labels for test set (compressed npz/parquet)
- model checkpoint + config snapshot
- git commit hash recorded in `metrics.json` for traceability

### agent notes (write here)




---

## 6) development plan (recursive testing loop, git discipline)

### branching
- `main` protected by tests
- feature branches per milestone or short-lived branches
- every merge must pass CI

### commit discipline
- small commits that keep the repo green
- commit messages include: scope, what tests were added, what passed

### recursive loop (mandatory for every milestone)
1) write/update spec (contracts, success criteria, acceptance gates)
2) add failing unit/integration test(s)
3) implement minimal code to pass
4) run tests locally
5) commit
6) re-run full suite (unit + smoke) as regression
7) only then start next milestone

### commands you will use repeatedly (adjust to repo tooling)
```bash
pytest -q
pytest -q -m "smoke"
git status
git add -A
git commit -m "feat(data): subject-wise split + leakage tests"
```

---

## 7) past-commit retesting rule (regression discipline)

After implementing any new model or changing preprocessing:

- re-run the smoke suite
- re-run at least one historical "golden" run test
- confirm calibration + HMM artifacts are still produced

### agent notes (write here)




---

## 8) milestones (tests first, then minimal implementation)

### milestone 0: skeleton + CI + fixtures

Tests to write first:
- test framework boots and discovers tests
- smoke fixture dataset loader returns deterministic sample
- CLI "help" returns 0

Implementation:
- package structure, config loader, logging, artifact directory conventions
- minimal CLI scaffolding

Commit gate:
- all tests green in CI

### milestone 1: ingest + preprocessing + subject-wise splits

Tests:
- subject-wise split has zero subject overlap
- label mapping covers exactly {W,N1,N2,N3,REM} and rejects unknowns
- epoch normalization/filtering deterministic given seed/config

Implementation:
- dataset adapter interface
- preprocessing pipeline with explicit, versioned parameters

Commit gate:
- smoke run completes through split stage, writes split manifest

### milestone 2: metrics + calibration module (before any model work)

Tests:
- macro F1 and confusion matrix correct on toy arrays
- ECE and Brier correct on toy probabilities
- probability simplex checks

Implementation:
- eval package exports metrics.json and plots
- strict schema for metrics.json

Commit gate:
- smoke run completes with dummy predictions and exports all artifacts

### milestone 3: classical ML baseline (engineered features)

Tests:
- feature extraction expected shape + finite values
- features stable across runs for same input
- baseline outputs well-formed probabilities

Implementation:
- time-domain + frequency-domain descriptors (document exactly which)
- RF/SVM/LR via scikit-learn pipelines
- save model and per-epoch probabilities

Commit gate:
- smoke run trains baseline and produces full artifact set

### milestone 4: 1D CNN on raw/minimally processed epochs

Tests:
- forward pass shape and simplex
- training step decreases loss on tiny synthetic task
- deterministic inference under fixed seed where feasible

Implementation:
- small CNN with parameter count reporting
- training loop, checkpointing, config logging

Commit gate:
- baseline and CNN runnable via config switch; regressions pass

### milestone 5: context-aware CNN-BiLSTM

Tests:
- sequence windowing alignment correct
- padding/masking correct
- outputs aligned to correct epochs

Implementation:
- CNN encoder per epoch + BiLSTM over windows
- boundary handling documented

Commit gate:
- end-to-end runs, metrics produced, previous models unaffected

### milestone 6: attention-based seq model (SeqSleepNet-style)

Tests:
- attention weights shape sanity
- masking/alignment correct
- overfit test on tiny subset

Implementation:
- GRU encoder-decoder with attention over context window
- optional attention summary artifacts (interpretability)

Commit gate:
- all previous tests and golden runs still pass

### milestone 7: calibration methods (post-hoc)

Tests:
- temperature scaling reduces NLL on held-out calibration split in controlled toy case
- isotonic/Platt (if used) produces valid probabilities and expected constraints

Implementation:
- calibration wrapper for any model outputs
- calibration artifacts always generated

Commit gate:
- calibrated probs saved and evaluated for all model types

### milestone 8: HMM post-processing

Tests:
- transition matrix rows sum to 1
- Viterbi decode length matches input
- smoothing reduces implausible transitions on crafted sequence without destroying labels
- guardrail: does not reduce macro F1 beyond tolerance on smoke

Implementation:
- transition matrix from training labels (optional) plus physiology-informed priors (documented)
- Viterbi on log-probabilities
- outputs both raw and smoothed predictions

Commit gate:
- report compares raw vs smoothed metrics; artifacts include both

### agent notes (write here)




---

## 9) specification details to write early (and version)

### data spec
- spec file: `specs/dataset_sleep_edfx_st.yaml` with a version field
- dataset subset: Sleep-EDFx v1.0.0 sleep-telemetry (ST*) only
- epoch length 30 s; sampling rate 100 Hz (native, no resampling in pinned protocol)
- channel policy: default EEG Fpz-Cz; optional Pz-Oz as ablation
- label source + mapping: W->W, 1->N1, 2->N2, 3+4->N3, R->REM; exclude M and ? and log removals
- subject identifiers and night grouping (both nights of a subject stay together)
- split strategy: headline LOSO by subject; optional 5-fold subject-wise CV for secondary reporting

### model spec (per model family)
- exact input tensor shapes
- outputs: logits and probabilities for 5 classes
- training objective, class imbalance handling (if any)
- parameter count and inference speed measurement method

### evaluation spec
- spec file: `specs/evaluation.yaml` with protocol_version and seed
- headline protocol: LOSO by subject; both nights held out together
- secondary protocol (optional): 5-fold subject-wise CV
- metrics primary: macro F1; macro-averaging definition pinned
- calibration: 80/20 subject-wise split within training fold; ECE with 15 equal-width bins
- optional confidence interval method

### HMM transition rules spec
- spec file: `specs/hmm_transition_rules.yaml`
- define allowed transitions or a transition prior matrix, plus smoothing weight strategy
- define implausible transitions for reporting:
  - N3 -> REM and REM -> N3
  - W -> N3 and N3 -> W
- HMM may still allow low-probability transitions; rules are primarily for reporting and guardrails

### artifact schema contracts
- spec file: `specs/artifacts_schema.yaml`
- `metrics.json` required fields:
  - run_id, git_commit, timestamp_utc
  - dataset_name, dataset_version, dataset_subset, channel_config
  - split_protocol, protocol_version, fold_id
  - model_name, model_version, param_count
  - train_time_sec, infer_epochs_per_sec
  - metrics: macro_f1, accuracy, balanced_accuracy
  - per_class_f1 (W,N1,N2,N3,REM) and confusion_matrix (counts + class order)
  - calibration: ece, brier, nll, n_bins, calibration_method
  - temporal: implausible_transition_rate_raw, implausible_transition_rate_hmm
  - artifacts: paths to generated files
- per-subject CSV columns:
  - subject_id, n_epochs
  - macro_f1, accuracy, balanced_accuracy
  - f1_W, f1_N1, f1_N2, f1_N3, f1_REM
  - ece, brier, nll (if computed per subject)
  - implausible_transition_rate_raw, implausible_transition_rate_hmm
- predictions layout per run:
  - `predictions/raw_probs` (npz or parquet): subject_id, epoch_index, probs[5]
  - `predictions/raw_pred_labels` (csv): subject_id, epoch_index, label
  - `predictions/hmm_pred_labels` (csv): subject_id, epoch_index, label

### CLI contract
- config format: YAML
- commands:
  - `sleepstage fetch-data --dest data/raw/sleep-edfx-1.0.0`
  - `sleepstage make-fixture --n-pairs 10 --out data/fixtures/smoke`
  - `sleepstage train --config configs/model.yaml`
  - `sleepstage eval --run runs/<run_id>`
- CLI must return nonzero on missing data/config and write a run directory with `metrics.json`

### agent notes (write here)




---

## 10) operational requirements (reproducibility, traceability)

Every run must record:
- git commit hash
- full config
- package versions (lockfile + snapshot)
- random seeds
- hardware summary (cpu/gpu)

Determinism policy:
- document what is deterministic vs best-effort
- tests avoid brittle exact matches unless CPU + controlled seeds
- one seed stored in config; derive and log seeds for random, numpy, and torch
- set `PYTHONHASHSEED`, `random.seed`, `numpy.random.seed`, `torch.manual_seed`, and `torch.cuda.manual_seed_all` (if CUDA)
- CPU smoke runs use `torch.use_deterministic_algorithms(True)` where supported
- CUDA determinism is best-effort: set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`
- if an op is nondeterministic or unsupported, document the exception and the tolerance used in reproducibility gates

### agent notes (write here)




---

## 11) final human verification checklist (complete before any public submission)

- success criteria thresholds explicitly written
- evaluation split is subject-wise and validated by tests
- no clinical claims beyond measured metrics
- external dataset references accurate and human-verified
- all unit and integration tests pass on a clean machine
- documentation discloses AI assistance and human verification
- release artifacts include commit hash and configs

### agent notes (write here)




---

## 12) agent marching orders (non-negotiable)

Start by implementing milestone 0-2 (skeleton, data provisioning, subject-wise splits, metrics/calibration computation) with tests first. Do not implement deep models until metrics, calibration metrics, artifact export, and the smoke pipeline exist and are tested. Keep the repo green, commit frequently, and re-run regression smoke runs after every milestone.

### agent notes (write here)
