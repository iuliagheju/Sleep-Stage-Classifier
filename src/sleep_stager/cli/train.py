"""Typer CLI entry-point for training + evaluation runs."""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import typer
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ..data.dataset import flatten_subjects
from ..data.ingest import DataConfig, load_subjects, stage_to_index_map
from ..data.preprocessing import PreprocessConfig, normalize_epochs
from ..data.splits import (
    SplitConfig,
    dump_splits,
    subject_wise_kfold_splits,
    subject_wise_split,
    unique_subject_ids,
)
from ..eval.calibration import (
    CalibrationConfig,
    brier_score,
    dump_calibration_metrics,
    expected_calibration_error,
    reliability_diagram,
)
from ..eval.baseline import compute_classical_baseline_metrics
from ..eval.gates import evaluate_gates
from ..eval.metrics import (
    MetricConfig,
    compute_classification_metrics,
    confusion_matrix_artifacts,
    dump_metrics,
    per_class_f1,
    per_subject_metrics,
)
from ..eval.schema import validate_metrics_schema
from ..eval.temporal import (
    TransitionRule,
    estimate_transition_matrix,
    implausible_transition_rate,
    per_subject_implausible_rates,
    stage_change_rate,
    sequences_by_subject,
)
from ..features.bandpower import BandConfig
from ..features.transforms import FeatureConfig, build_feature_matrix, build_feature_names
from ..models import calibration as calib
from ..models import attention, classical, cnn, seq
from ..postprocess.hmm import HMMConfig, smooth_sequence
from ..utils.logging import configure_logging
from ..utils.random import seed_everything
from ..utils.system import collect_hardware_summary, collect_package_versions

ROOT = Path(__file__).resolve().parents[3]
app = typer.Typer(add_completion=False)
ALLOWED_PROTOCOLS = {"loso", "kfold", "kfold_subject", "kfold-subject"}
K_FOLD_PROTOCOLS = {"kfold", "kfold_subject", "kfold-subject"}


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _resolve_config(config_path: str, config_name: str, overrides: List[str]):
    config_dir = Path(config_path).resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_protocol(eval_spec: dict, protocol_override: str | None) -> str:
    if protocol_override:
        protocol = str(protocol_override).lower()
        if protocol == "secondary":
            protocol = str(eval_spec.get("secondary_protocol", {}).get("name", "kfold_subject")).lower()
        if protocol == "headline":
            protocol = str(eval_spec.get("headline_protocol", {}).get("name", "loso")).lower()
    else:
        protocol = str(eval_spec.get("headline_protocol", {}).get("name", "loso")).lower()
    if protocol not in ALLOWED_PROTOCOLS:
        allowed = ", ".join(sorted(ALLOWED_PROTOCOLS))
        raise ValueError(f"Unsupported evaluation.protocol '{protocol}'. Allowed: {allowed}")
    return protocol


def _select_kfold_fold(splits: List[Dict[str, List[str]]], fold_index: int) -> Dict[str, List[str]]:
    if fold_index < 0 or fold_index >= len(splits):
        raise ValueError(f"fold_index must be between 0 and {len(splits) - 1}")
    return splits[fold_index]


def _align_probs(probs: np.ndarray, class_order: Iterable[str], target_order: Iterable[str]) -> np.ndarray:
    class_to_idx = {label: idx for idx, label in enumerate(class_order)}
    aligned = np.zeros((probs.shape[0], len(list(target_order))))
    for tgt_idx, label in enumerate(target_order):
        if label not in class_to_idx:
            continue
        aligned[:, tgt_idx] = probs[:, class_to_idx[label]]
    row_sum = aligned.sum(axis=1, keepdims=True) + 1e-9
    return aligned / row_sum


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _git_dirty() -> bool:
    try:
        output = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=ROOT)
    except Exception:  # pragma: no cover
        return False
    return bool(output.strip())


def _hash_payload(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _mask(subject_ids: np.ndarray, subset: Iterable[str]) -> np.ndarray:
    return np.isin(subject_ids, list(subset))


def _collect_deep_probs(
    model, signals: np.ndarray, batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    import torch

    if signals.shape[0] == 0:
        return np.array([]), np.zeros((0, 0))
    all_probs = []
    with torch.no_grad():
        for start in range(0, signals.shape[0], batch_size):
            batch = signals[start : start + batch_size]
            tensor = torch.tensor(batch, dtype=torch.float32)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def _labels_from_indices(indices: np.ndarray, idx_to_label: Dict[int, str]) -> np.ndarray:
    return np.array([idx_to_label[idx] for idx in indices])


def _dump_confusion_png(matrix: np.ndarray, labels: Iterable[str], path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


@app.command()
def train(
    config_path: str = typer.Option("configs", help="Directory containing Hydra configs"),
    config_name: str = typer.Option("default", help="Config file name inside config_path"),
    override: List[str] = typer.Option(
        [],
        "--override",
        help="Hydra-style override, e.g. --override model.family=classical",
    ),
) -> None:
    cfg = _resolve_config(config_path, config_name, override)
    dataset_spec = _load_yaml(ROOT / "specs" / "dataset_sleep_edfx_st.yaml")
    eval_spec = _load_yaml(ROOT / "specs" / "evaluation.yaml")
    hmm_spec = _load_yaml(ROOT / "specs" / "hmm_transition_rules.yaml")
    artifact_schema = _load_yaml(ROOT / "specs" / "artifacts_schema.yaml")
    model_limits = _load_yaml(ROOT / "specs" / "model_limits.yaml")
    spec_hash = _hash_payload(
        {
            "dataset_spec": dataset_spec,
            "evaluation_spec": eval_spec,
            "hmm_transition_rules": hmm_spec,
            "artifacts_schema": artifact_schema,
            "model_limits": model_limits,
        }
    )

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    artifact_root = Path(cfg.artifacts.root)
    run_dir = artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir / "logs")
    git_commit = _git_sha()
    git_dirty = _git_dirty()
    seed_cfg = seed_everything()
    hardware = collect_hardware_summary()
    packages = collect_package_versions(
        [
            "numpy",
            "scipy",
            "pandas",
            "scikit-learn",
            "torch",
            "mne",
        ]
    )

    data_cfg = DataConfig(
        data_dir=Path(cfg.data.dir),
        channel=str(getattr(cfg.data, "channel", "Fpz-Cz")),
        epoch_length_sec=int(getattr(cfg.data, "epoch_length_sec", 30)),
        sample_rate=int(getattr(cfg.data, "sample_rate", 100)),
    )
    subjects = list(load_subjects(data_cfg))
    pre_cfg = PreprocessConfig(**OmegaConf.to_container(cfg.preprocess, resolve=True))
    for subject in subjects:
        subject.signals = normalize_epochs(subject.signals, pre_cfg)

    band_cfg = BandConfig(sample_rate=int(cfg.bandpower.sample_rate))
    feature_cfg = FeatureConfig(**OmegaConf.to_container(cfg.features, resolve=True))
    X, y, subject_index = build_feature_matrix(subjects, band_cfg, feature_cfg)
    signals, labels, subject_index_raw = flatten_subjects(subjects)
    feature_names = build_feature_names(signals.shape[1], band_cfg, feature_cfg)
    feature_dim = int(X.shape[1])
    if feature_names and len(feature_names) != feature_dim:
        feature_names = []
    assert np.array_equal(subject_index, subject_index_raw)

    split_cfg = SplitConfig()
    subject_ids_all = sorted(unique_subject_ids([sub.subject_id for sub in subjects]))
    allow_single = bool(getattr(cfg.evaluation, "allow_single_subject", False))
    protocol_override = getattr(cfg.evaluation, "protocol", None)
    protocol = _resolve_protocol(eval_spec, protocol_override)
    fold_id = None
    split_protocol_name = protocol

    if len(subject_ids_all) < 2:
        if allow_single:
            logger.warning("Only one subject found; using train=test for smoke evaluation.")
            splits = {"train": subject_ids_all, "val": [], "test": subject_ids_all}
            if protocol in K_FOLD_PROTOCOLS:
                fold_id = 0
        else:
            raise ValueError(
                "Need at least 2 subjects for subject-wise evaluation. Set evaluation.allow_single_subject=true to override."
            )
    else:
        if protocol in K_FOLD_PROTOCOLS:
            k = int(eval_spec.get("secondary_protocol", {}).get("k", 5))
            cal_ratio = float(eval_spec.get("calibration_split", {}).get("calibration_ratio", split_cfg.val_ratio))
            splits_all = subject_wise_kfold_splits(subject_ids_all, k, split_cfg.seed, val_ratio=cal_ratio)
            fold_index = int(getattr(cfg.evaluation, "fold_index", 0))
            splits = _select_kfold_fold(splits_all, fold_index)
            fold_id = fold_index
        else:
            splits = subject_wise_split(subject_ids_all, split_cfg)
            if protocol == "loso":
                test_subject = subject_ids_all[-1]
                train_val_subjects = subject_ids_all[:-1]
                cal_ratio = float(eval_spec.get("calibration_split", {}).get("calibration_ratio", 0.2))
                val_count = max(1, int(len(train_val_subjects) * cal_ratio)) if len(train_val_subjects) > 1 else 0
                val_subjects = train_val_subjects[-val_count:] if val_count > 0 else []
                train_subjects = train_val_subjects[:-val_count] if val_count > 0 else train_val_subjects
                splits = {"train": train_subjects, "val": val_subjects, "test": [test_subject]}
    dump_splits(splits, run_dir / "splits.json")

    train_mask = _mask(subject_index, splits["train"])
    val_mask = _mask(subject_index, splits["val"])
    test_mask = _mask(subject_index, splits["test"])
    train_mask_raw = _mask(subject_index_raw, splits["train"])
    val_mask_raw = _mask(subject_index_raw, splits["val"])
    test_mask_raw = _mask(subject_index_raw, splits["test"])

    label_to_idx = stage_to_index_map(data_cfg.stage_labels)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    stages = list(label_to_idx.keys())

    model_family = cfg.model.family
    logger.info(f"Training model family: {model_family}")
    train_start = time.perf_counter()

    if not train_mask.any():
        raise ValueError("Training split is empty. Adjust split configuration or dataset size.")

    def _empty_probs(num_classes: int) -> np.ndarray:
        return np.zeros((0, num_classes))

    baseline_metrics = None
    baseline_macro_f1 = getattr(cfg.evaluation, "baseline_macro_f1", None)
    compute_baseline = bool(getattr(cfg.evaluation, "compute_baseline", False))
    baseline_cfg = None
    if compute_baseline and model_family != "classical":
        baseline_cfg = classical.ClassicalConfig(**OmegaConf.to_container(cfg.model.classical, resolve=True))
        baseline_metrics = compute_classical_baseline_metrics(
            X[train_mask],
            y[train_mask],
            X[test_mask],
            y[test_mask],
            baseline_cfg,
            stages,
        )
        baseline_macro_f1 = float(baseline_metrics.get("macro_f1", 0.0))

    if model_family == "classical":
        cls_cfg = classical.ClassicalConfig(**OmegaConf.to_container(cfg.model.classical, resolve=True))
        model = classical.train_model(X[train_mask], y[train_mask], cls_cfg)
        train_end = time.perf_counter()
        train_preds, train_probs_raw = classical.predict(model, X[train_mask])
        if val_mask.any():
            val_preds, val_probs_raw = classical.predict(model, X[val_mask])
        else:
            val_preds = np.array([])
            val_probs_raw = _empty_probs(len(model.classes_))
        infer_start = time.perf_counter()
        if test_mask.any():
            test_preds, test_probs_raw = classical.predict(model, X[test_mask])
        else:
            test_preds = np.array([])
            test_probs_raw = _empty_probs(len(model.classes_))
        infer_end = time.perf_counter()
        train_probs = _align_probs(train_probs_raw, model.classes_, stages)
        val_probs = _align_probs(val_probs_raw, model.classes_, stages)
        test_probs = _align_probs(test_probs_raw, model.classes_, stages)
        test_labels = np.array(test_preds)
    elif model_family == "cnn":
        cnn_cfg = cnn.CNNConfig(**OmegaConf.to_container(cfg.model.cnn, resolve=True))
        model, _ = cnn.train_model(signals[train_mask_raw], labels[train_mask_raw], label_to_idx, cnn_cfg)
        train_end = time.perf_counter()
        train_idx, train_probs = _collect_deep_probs(model, signals[train_mask_raw], cnn_cfg.batch_size)
        if val_mask_raw.any():
            val_idx, val_probs = _collect_deep_probs(model, signals[val_mask_raw], cnn_cfg.batch_size)
        else:
            val_idx = np.array([])
            val_probs = _empty_probs(len(stages))
        infer_start = time.perf_counter()
        if test_mask_raw.any():
            test_idx, test_probs = _collect_deep_probs(model, signals[test_mask_raw], cnn_cfg.batch_size)
        else:
            test_idx = np.array([])
            test_probs = _empty_probs(len(stages))
        infer_end = time.perf_counter()
        train_preds = _labels_from_indices(train_idx, idx_to_label)
        val_preds = _labels_from_indices(val_idx, idx_to_label)
        test_labels = _labels_from_indices(test_idx, idx_to_label)
    elif model_family == "attention":
        attn_cfg = attention.AttentionConfig(**OmegaConf.to_container(cfg.model.attention, resolve=True))
        model, _ = attention.train_model(signals[train_mask_raw], labels[train_mask_raw], label_to_idx, attn_cfg)
        train_end = time.perf_counter()
        train_idx, train_probs = attention.predict(model, signals[train_mask_raw])
        if val_mask_raw.any():
            val_idx, val_probs = attention.predict(model, signals[val_mask_raw])
        else:
            val_idx = np.array([])
            val_probs = _empty_probs(len(stages))
        infer_start = time.perf_counter()
        if test_mask_raw.any():
            test_idx, test_probs = attention.predict(model, signals[test_mask_raw])
        else:
            test_idx = np.array([])
            test_probs = _empty_probs(len(stages))
        infer_end = time.perf_counter()
        train_preds = _labels_from_indices(train_idx, idx_to_label)
        val_preds = _labels_from_indices(val_idx, idx_to_label)
        test_labels = _labels_from_indices(test_idx, idx_to_label)
    else:
        seq_cfg = seq.SeqConfig(**OmegaConf.to_container(cfg.model.seq, resolve=True))
        model, _ = seq.train_model(signals[train_mask_raw], labels[train_mask_raw], label_to_idx, seq_cfg)
        train_end = time.perf_counter()
        train_idx, train_probs = seq.predict(model, signals[train_mask_raw])
        if val_mask_raw.any():
            val_idx, val_probs = seq.predict(model, signals[val_mask_raw])
        else:
            val_idx = np.array([])
            val_probs = _empty_probs(len(stages))
        infer_start = time.perf_counter()
        if test_mask_raw.any():
            test_idx, test_probs = seq.predict(model, signals[test_mask_raw])
        else:
            test_idx = np.array([])
            test_probs = _empty_probs(len(stages))
        infer_end = time.perf_counter()
        train_preds = _labels_from_indices(train_idx, idx_to_label)
        val_preds = _labels_from_indices(val_idx, idx_to_label)
        test_labels = _labels_from_indices(test_idx, idx_to_label)

    train_time = (train_end - train_start) if "train_end" in locals() else time.perf_counter() - train_start

    if test_probs.shape[0] == 0:
        raise ValueError("Test split is empty. Adjust split configuration or dataset size.")

    test_probs_raw = test_probs.copy()
    if cfg.calibration.enabled and val_probs.shape[0] > 0:
        temp_cfg = calib.TemperatureConfig(**OmegaConf.to_container(cfg.calibration.temperature, resolve=True))
        val_targets = np.array([label_to_idx[label] for label in labels[val_mask_raw]])
        scaler = calib.fit_temperature(np.log(np.clip(val_probs, 1e-9, 1.0)), val_targets, temp_cfg)
        test_probs = calib.apply_temperature(scaler, np.log(np.clip(test_probs, 1e-9, 1.0)))
    metric_cfg = MetricConfig(labels=stages)
    base_metrics = compute_classification_metrics(np.array(labels[test_mask_raw]), test_labels, metric_cfg)
    per_class = per_class_f1(np.array(labels[test_mask_raw]), test_labels, stages)
    logger.info(f"Test macro F1 (pre-HMM): {base_metrics['macro_f1']:.3f}")

    transition_prior = hmm_spec.get("transition_prior", {})
    prior_weight = float(transition_prior.get("prior_weight", 0.0))
    min_transition_prob = float(transition_prior.get("min_transition_prob", cfg.hmm.min_prob))
    empirical_transition = estimate_transition_matrix(
        labels[train_mask_raw].tolist(),
        stages,
        prior_weight=prior_weight,
        min_prob=min_transition_prob,
    )

    implausible_rules = [
        TransitionRule(source=rule["from"], target=rule["to"]) for rule in hmm_spec.get("implausible_transitions", [])
    ]
    implausible_raw = implausible_transition_rate(test_labels.tolist(), implausible_rules)

    hmm_metrics = base_metrics
    smoothed = test_labels.tolist()
    if cfg.hmm.enabled:
        hmm_cfg = HMMConfig(
            states=stages,
            transition_bias=float(cfg.hmm.transition_bias),
            min_prob=float(cfg.hmm.min_prob),
            prior_weight=prior_weight,
        )
        smoothed = smooth_sequence(
            test_probs,
            label_to_idx,
            hmm_cfg,
            transition=empirical_transition,
        )
        hmm_metrics = compute_classification_metrics(
            np.array(labels[test_mask_raw]),
            np.array(smoothed),
            metric_cfg,
        )
        implausible_hmm = implausible_transition_rate(smoothed, implausible_rules)
        drop = base_metrics["macro_f1"] - hmm_metrics["macro_f1"]
        reduction = implausible_raw - implausible_hmm
        guard_drop = float(eval_spec.get("gates", {}).get("hmm", {}).get("macro_f1_drop_max", cfg.evaluation.hmm_guardrail))
        min_reduction = float(eval_spec.get("gates", {}).get("hmm", {}).get("implausible_transition_reduction_min", 0.0))
        if drop > guard_drop or reduction < min_reduction:
            logger.warning("HMM guardrail triggered; reverting to pre-HMM predictions")
            hmm_metrics = base_metrics
            smoothed = test_labels.tolist()
            implausible_hmm = implausible_raw
    else:
        implausible_hmm = implausible_raw

    calib_cfg = CalibrationConfig(num_bins=int(cfg.evaluation.ece_bins))
    per_subject = per_subject_metrics(subject_index[test_mask], np.array(labels[test_mask_raw]), test_labels, stages)
    per_subj_raw_impl = per_subject_implausible_rates(subject_index[test_mask], np.array(test_labels), implausible_rules)
    per_subj_hmm_impl = per_subject_implausible_rates(subject_index[test_mask], np.array(smoothed), implausible_rules)
    per_subject_path = run_dir / "per_subject_metrics.csv"
    rows = []
    for subj, metrics_row in per_subject.items():
        mask = subject_index[test_mask] == subj
        n_epochs = int(mask.sum())
        if n_epochs == 0:
            continue
        y_true_sub = np.array(labels[test_mask_raw])[mask]
        y_pred_sub = np.array(test_labels)[mask]
        probs_sub = test_probs[mask]
        y_true_idx_sub = np.array([label_to_idx[l] for l in y_true_sub])
        ece_sub = expected_calibration_error(y_true_idx_sub, probs_sub, calib_cfg)
        brier_sub = brier_score(y_true_idx_sub, probs_sub, len(stages))
        nll_sub = float(-np.log(np.clip(probs_sub[np.arange(len(probs_sub)), y_true_idx_sub], 1e-12, 1.0)).mean())
        per_label_f1 = per_class_f1(y_true_sub, y_pred_sub, stages)
        row = {
            "subject_id": subj,
            "n_epochs": n_epochs,
            "macro_f1": metrics_row.get("macro_f1"),
            "accuracy": metrics_row.get("accuracy"),
            "balanced_accuracy": metrics_row.get("balanced_accuracy", metrics_row.get("accuracy")),
            "ece": ece_sub,
            "brier": brier_sub,
            "nll": nll_sub,
            "implausible_transition_rate_raw": per_subj_raw_impl.get(subj, 0.0),
            "implausible_transition_rate_hmm": per_subj_hmm_impl.get(subj, 0.0),
        }
        for label in stages:
            row[f"f1_{label}"] = per_label_f1.get(label, 0.0)
        rows.append(row)
    per_subj_df = pd.DataFrame(rows)
    per_subj_df.to_csv(per_subject_path, index=False)

    config_dump = OmegaConf.to_container(cfg, resolve=True)
    config_hash = _hash_payload(config_dump)
    y_true_idx = np.array([label_to_idx[label] for label in labels[test_mask_raw]])
    ece_before = float(expected_calibration_error(y_true_idx, test_probs_raw, calib_cfg))
    brier_before = float(brier_score(y_true_idx, test_probs_raw, len(stages)))
    nll_before = float(
        -np.log(np.clip(test_probs_raw[np.arange(len(test_probs_raw)), y_true_idx], 1e-12, 1.0)).mean()
    )
    ece = float(expected_calibration_error(y_true_idx, test_probs, calib_cfg))
    brier = float(brier_score(y_true_idx, test_probs, len(stages)))
    nll = float(-np.log(np.clip(test_probs[np.arange(len(test_probs)), y_true_idx], 1e-12, 1.0)).mean())
    calibration_metrics = {
        "ece": ece,
        "brier": brier,
        "nll": nll,
        "n_bins": calib_cfg.num_bins,
        "ece_before": ece_before,
        "brier_before": brier_before,
        "nll_before": nll_before,
    }
    dump_calibration_metrics(calibration_metrics, run_dir / "calibration.json")
    reliability_diagram(y_true_idx, test_probs, calib_cfg, run_dir / "calibration.png")

    cm = confusion_matrix_artifacts(np.array(labels[test_mask_raw]), test_labels, stages)
    _dump_confusion_png(cm["matrix"], cm["labels"], run_dir / "confusion_matrix.png")
    (run_dir / "confusion_matrix.json").write_text(json.dumps({"labels": cm["labels"], "matrix": cm["matrix"].tolist()}, indent=2))

    pred_payload = {
        "subject_id": subject_index[test_mask],
        "epoch_index": np.arange(test_probs.shape[0]),
    }
    for idx, label in enumerate(stages):
        pred_payload[f"prob_{label}"] = test_probs[:, idx]
    probs_df = pd.DataFrame(pred_payload)
    probs_path = run_dir / "predictions_raw_probs.parquet"
    probs_df.to_parquet(probs_path, index=False)

    raw_labels_df = pd.DataFrame(
        {
            "subject_id": subject_index[test_mask],
            "epoch_index": np.arange(test_probs.shape[0]),
            "label": test_labels,
        }
    )
    raw_labels_path = run_dir / "predictions_raw_labels.csv"
    raw_labels_df.to_csv(raw_labels_path, index=False)

    hmm_labels_df = pd.DataFrame(
        {
            "subject_id": subject_index[test_mask],
            "epoch_index": np.arange(test_probs.shape[0]),
            "label": smoothed,
        }
    )
    hmm_labels_path = run_dir / "predictions_hmm_labels.csv"
    hmm_labels_df.to_csv(hmm_labels_path, index=False)

    infer_epochs_per_sec = test_probs.shape[0] / max((infer_end - infer_start), 1e-6)
    model_checkpoint = None
    if model_family == "classical":
        import joblib

        clf = getattr(model, "named_steps", {}).get("clf", model)
        coef = getattr(clf, "coef_", None)
        intercept = getattr(clf, "intercept_", None)
        if coef is not None and intercept is not None:
            param_count = int(coef.size + intercept.size)
        else:
            param_count = int(getattr(clf, "n_features_in_", 0))
        model_checkpoint = run_dir / "model.pkl"
        joblib.dump(model, model_checkpoint)
    else:
        param_count = int(sum(p.numel() for p in model.parameters()))
        import torch

        model_checkpoint = run_dir / "model.pt"
        torch.save(model.state_dict(), model_checkpoint)

    metrics_payload = {
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash": config_hash,
        "spec_hash": spec_hash,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "dataset_name": dataset_spec.get("dataset", {}).get("name", "unknown"),
        "dataset_version": dataset_spec.get("dataset", {}).get("version", "unknown"),
        "dataset_subset": dataset_spec.get("dataset", {}).get("subset", "unknown"),
        "channel_config": dataset_spec.get("signals", {}).get("eeg_channels", {}).get("default", ""),
        "split_protocol": split_protocol_name,
        "protocol_version": eval_spec.get("protocol_version", 1),
        "fold_id": fold_id,
        "model_name": model_family,
        "model_version": "0.1",
        "param_count": param_count,
        "train_time_sec": train_time,
        "infer_epochs_per_sec": infer_epochs_per_sec,
        "environment": {
            "seeds": {
                "numpy": seed_cfg.numpy_seed,
                "torch": seed_cfg.torch_seed,
                "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
            },
            "hardware": hardware,
            "packages": packages,
        },
        "features": {
            "use_bandpower": bool(feature_cfg.use_bandpower),
            "use_time_domain": bool(feature_cfg.use_time_domain),
            "use_bandpower_ratios": bool(feature_cfg.use_bandpower_ratios),
            "use_spectral_entropy": bool(feature_cfg.use_spectral_entropy),
            "feature_dim": feature_dim,
            "feature_names": feature_names,
        },
        "metrics": {
            **base_metrics,
            "hmm_macro_f1": hmm_metrics.get("macro_f1", base_metrics.get("macro_f1")),
            "per_class_f1": per_class,
            "confusion_matrix": {
                "class_order": stages,
                "counts": cm["matrix"].tolist(),
            },
        },
        "calibration": {
            **calibration_metrics,
            "calibration_method": "temperature",
        },
        "temporal": {
            "implausible_transition_rate_raw": implausible_raw,
            "implausible_transition_rate_hmm": implausible_hmm,
            "stage_changes_per_hour_raw": stage_change_rate(test_labels.tolist()),
            "stage_changes_per_hour_hmm": stage_change_rate(smoothed),
        },
        "artifacts": {},
        "config": config_dump,
    }
    if baseline_metrics is not None and baseline_cfg is not None:
        metrics_payload["baseline"] = {
            "model_type": baseline_cfg.model_type,
            **baseline_metrics,
        }
    gate_errors = evaluate_gates(metrics_payload, eval_spec, model_limits, baseline_macro_f1)
    metrics_payload["gates"] = {"passed": not gate_errors, "errors": gate_errors}
    if getattr(cfg.evaluation, "enforce_gates", False) and gate_errors:
        raise ValueError("Evaluation gates failed:\n" + "\n".join(gate_errors))
    dump_metrics(metrics_payload, run_dir / "metrics.json")

    metrics_payload["artifacts"] = {
        "per_subject_csv": str(per_subject_path),
        "confusion_matrix_png": str(run_dir / "confusion_matrix.png"),
        "confusion_matrix_json": str(run_dir / "confusion_matrix.json"),
        "reliability_diagram_png": str(run_dir / "calibration.png"),
        "calibration_json": str(run_dir / "calibration.json"),
        "predictions_raw_probs": str(probs_path),
        "predictions_raw_pred_labels": str(raw_labels_path),
        "predictions_hmm_pred_labels": str(hmm_labels_path),
        "model_checkpoint": str(model_checkpoint) if model_checkpoint else "",
        "config_snapshot": str(run_dir / "config_snapshot.yaml"),
    }
    schema_fields = artifact_schema.get("metrics_json", {}).get("fields", {})
    if schema_fields:
        schema_errors = validate_metrics_schema(metrics_payload, schema_fields)
        if schema_errors:
            raise ValueError("metrics.json schema validation failed:\n" + "\n".join(schema_errors))
    dump_metrics(metrics_payload, run_dir / "metrics.json")

    with (run_dir / "config_snapshot.yaml").open("w") as fh:
        yaml.safe_dump(config_dump, fh)

    logger.info(f"Run artifacts written to {run_dir}")


if __name__ == "__main__":
    app()
