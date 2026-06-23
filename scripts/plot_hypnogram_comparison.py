#!/usr/bin/env python
"""Generate true vs predicted hypnogram comparison plots for a training run.

This script reconstructs ground-truth labels for the run's test subjects from the
source dataset, aligns them with saved predictions, and writes one plot per
subject comparing:
- True labels
- Raw model predictions
- HMM-smoothed predictions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sleep_stager.data.ingest import DataConfig, SubjectRecord, load_subjects  # noqa: E402


STAGE_ORDER = ["W", "REM", "N1", "N2", "N3"]
STAGE_TO_Y = {stage: len(STAGE_ORDER) - idx - 1 for idx, stage in enumerate(STAGE_ORDER)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot true vs predicted hypnograms for a run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory with prediction CSVs")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <run-dir>/hypnograms)",
    )
    parser.add_argument(
        "--subject-id",
        action="append",
        default=[],
        help="Optional subject ID filter (repeatable)",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output DPI")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Plot file format")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _required_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def _subject_epoch_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subject_epoch_idx"] = out.groupby("subject_id").cumcount()
    return out


def _collect_true_labels_for_subjects(data_cfg: DataConfig, wanted_subjects: Iterable[str]) -> Dict[str, np.ndarray]:
    wanted = set(wanted_subjects)
    result: Dict[str, np.ndarray] = {}
    for subject in load_subjects(data_cfg):
        if subject.subject_id in wanted:
            labels = np.asarray(subject.labels).astype(str)
            result[subject.subject_id] = labels
            if len(result) == len(wanted):
                break
    missing = sorted(wanted - set(result.keys()))
    if missing:
        raise ValueError(
            "Could not reconstruct labels for all requested subjects. Missing: " + ", ".join(missing)
        )
    return result


def _map_labels_to_numeric(labels: np.ndarray) -> np.ndarray:
    return np.array([STAGE_TO_Y.get(str(label), np.nan) for label in labels], dtype=float)


def _align_predictions_with_truth(
    pred_df: pd.DataFrame,
    true_by_subject: Dict[str, np.ndarray],
) -> pd.DataFrame:
    aligned = pred_df.copy()
    aligned["true_label"] = ""

    for sid, group in aligned.groupby("subject_id", sort=False):
        true_labels = true_by_subject[sid]
        n_pred = len(group)
        n_true = len(true_labels)
        if n_pred != n_true:
            raise ValueError(
                f"Label count mismatch for {sid}: predictions={n_pred}, true={n_true}. "
                "Cannot safely align hypnogram."
            )
        aligned.loc[group.index, "true_label"] = true_labels

    return aligned


def _plot_subject(
    subject_id: str,
    df_raw_subj: pd.DataFrame,
    df_hmm_subj: pd.DataFrame,
    epoch_length_sec: int,
    out_path: Path,
    dpi: int,
) -> None:
    x_hours = (df_raw_subj["subject_epoch_idx"].to_numpy(dtype=float) * float(epoch_length_sec)) / 3600.0

    y_true = _map_labels_to_numeric(df_raw_subj["true_label"].to_numpy())
    y_raw = _map_labels_to_numeric(df_raw_subj["label"].to_numpy())
    y_hmm = _map_labels_to_numeric(df_hmm_subj["label"].to_numpy())

    tick_positions = [STAGE_TO_Y[s] for s in STAGE_ORDER]
    series = [
        ("True", y_true, "black"),
        ("Predicted (raw)", y_raw, "#d95f02"),
        ("Predicted (HMM)", y_hmm, "#1f77b4"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, (name, y_vals, color) in zip(axes, series):
        ax.step(x_hours, y_vals, where="post", linewidth=1.4, color=color)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(STAGE_ORDER)
        ax.set_ylabel(name)
        ax.grid(axis="x", alpha=0.25)

    axes[-1].set_xlabel("Time (hours)")
    fig.suptitle(f"Hypnogram Comparison - {subject_id}", y=0.98)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (run_dir / "hypnograms")

    config_path = _required_file(run_dir / "config_snapshot.yaml")
    split_manifest_path = _required_file(run_dir / "split_manifest.json")
    raw_pred_path = _required_file(run_dir / "predictions_raw_labels.csv")
    hmm_pred_path = _required_file(run_dir / "predictions_hmm_labels.csv")

    cfg = _load_yaml(config_path)
    split_manifest = _load_json(split_manifest_path)

    raw_df = pd.read_csv(raw_pred_path)
    hmm_df = pd.read_csv(hmm_pred_path)

    required_cols = {"subject_id", "epoch_index", "label"}
    if not required_cols.issubset(raw_df.columns):
        raise ValueError(f"Raw predictions missing columns: {sorted(required_cols - set(raw_df.columns))}")
    if not required_cols.issubset(hmm_df.columns):
        raise ValueError(f"HMM predictions missing columns: {sorted(required_cols - set(hmm_df.columns))}")

    raw_df = _subject_epoch_index(raw_df)
    hmm_df = _subject_epoch_index(hmm_df)

    # Validate that raw and HMM files refer to the same rows in the same order.
    key_cols = ["subject_id", "epoch_index", "subject_epoch_idx"]
    if not raw_df[key_cols].equals(hmm_df[key_cols]):
        raise ValueError(
            "Raw and HMM prediction files are not aligned by subject/epoch. "
            "Please regenerate run artifacts before plotting."
        )

    test_subject_ids = list(split_manifest.get("test_subject_ids", []))
    if not test_subject_ids:
        raise ValueError("split_manifest.json has no test_subject_ids")

    selected_subjects = test_subject_ids
    if args.subject_id:
        requested = set(args.subject_id)
        selected_subjects = [sid for sid in test_subject_ids if sid in requested]
        missing = sorted(requested - set(selected_subjects))
        if missing:
            raise ValueError("Requested subject_id not present in test split: " + ", ".join(missing))

    raw_df = raw_df[raw_df["subject_id"].isin(selected_subjects)].reset_index(drop=True)
    hmm_df = hmm_df[hmm_df["subject_id"].isin(selected_subjects)].reset_index(drop=True)

    data_cfg = DataConfig(
        data_dir=Path(cfg["data"]["dir"]),
        channel=str(cfg["data"].get("channel", "Fpz-Cz")),
        epoch_length_sec=int(cfg["data"].get("epoch_length_sec", 30)),
        sample_rate=int(cfg["data"].get("sample_rate", 100)),
    )
    epoch_length_sec = int(cfg["data"].get("epoch_length_sec", 30))

    true_by_subject = _collect_true_labels_for_subjects(data_cfg, selected_subjects)
    raw_aligned = _align_predictions_with_truth(raw_df, true_by_subject)

    # Bring true labels into HMM DF using identical row keys.
    hmm_aligned = hmm_df.merge(
        raw_aligned[["subject_id", "epoch_index", "subject_epoch_idx", "true_label"]],
        on=["subject_id", "epoch_index", "subject_epoch_idx"],
        how="left",
    )
    if hmm_aligned["true_label"].isna().any():
        raise ValueError("Failed to align true labels onto HMM predictions")

    out_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in selected_subjects:
        df_raw_subj = raw_aligned[raw_aligned["subject_id"] == subject_id]
        df_hmm_subj = hmm_aligned[hmm_aligned["subject_id"] == subject_id]
        if len(df_raw_subj) == 0:
            continue
        out_path = out_dir / f"{subject_id}_hypnogram_compare.{args.format}"
        _plot_subject(
            subject_id=subject_id,
            df_raw_subj=df_raw_subj,
            df_hmm_subj=df_hmm_subj,
            epoch_length_sec=epoch_length_sec,
            out_path=out_path,
            dpi=args.dpi,
        )

    print(f"Saved {len(selected_subjects)} hypnogram plot(s) to: {out_dir}")


if __name__ == "__main__":
    main()
