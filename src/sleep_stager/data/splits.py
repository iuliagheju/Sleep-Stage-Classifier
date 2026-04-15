"""Subject-wise splitting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import json
import numpy as np


@dataclass(slots=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio


def subject_wise_split(subject_ids: List[str], config: SplitConfig) -> Dict[str, List[str]]:
    if config.train_ratio + config.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")
    subject_ids = unique_subject_ids(subject_ids)
    rng = np.random.default_rng(config.seed)
    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    train_count = max(1, int(n * config.train_ratio))
    if n >= 3:
        val_count = max(1, int(n * config.val_ratio)) if config.val_ratio > 0 else 0
    else:
        val_count = max(0, n - train_count - 1)
    remaining = n - train_count - val_count
    test_count = max(1 if n >= 3 else remaining, remaining)
    # adjust totals if we over-allocated
    while train_count + val_count + test_count > n:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            test_count -= 1
    while train_count + val_count + test_count < n:
        test_count += 1
    train_end = train_count
    val_end = train_end + val_count
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def subject_wise_train_val_split(
    subject_ids: Iterable[str], val_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be >= 0 and < 1")
    subjects = unique_subject_ids(subject_ids)
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return [], []
    if n == 1:
        return shuffled, []
    if val_ratio == 0:
        return shuffled, []
    val_count = max(1, int(n * val_ratio))
    val_count = min(val_count, n - 1)
    train = shuffled[:-val_count]
    val = shuffled[-val_count:]
    return train, val


def subject_wise_kfold_splits(
    subject_ids: Iterable[str], k: int, seed: int, val_ratio: float
) -> List[Dict[str, List[str]]]:
    if k < 2:
        raise ValueError("k must be >= 2 for k-fold splits")
    subjects = unique_subject_ids(subject_ids)
    if len(subjects) < k:
        raise ValueError("k-fold requires at least k subjects")
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    folds = [list(fold) for fold in np.array_split(shuffled, k)]
    splits: List[Dict[str, List[str]]] = []
    for fold_idx, test_fold in enumerate(folds):
        train_val = [sid for idx, fold in enumerate(folds) if idx != fold_idx for sid in fold]
        train, val = subject_wise_train_val_split(train_val, val_ratio, seed + fold_idx + 1)
        splits.append({"train": train, "val": val, "test": list(test_fold)})
    return splits


def subject_wise_loso_split(
    subject_ids: Iterable[str],
    val_ratio: float,
    seed: int,
    held_out_subject_id: str | None = None,
) -> Tuple[Dict[str, List[str]], str]:
    subjects = unique_subject_ids(subject_ids)
    if len(subjects) < 2:
        raise ValueError("LOSO requires at least 2 subjects")
    if held_out_subject_id is not None:
        if held_out_subject_id not in subjects:
            raise ValueError(f"Held-out subject '{held_out_subject_id}' not in dataset")
        test_subject = held_out_subject_id
        train_val_subjects = [sid for sid in subjects if sid != held_out_subject_id]
    else:
        test_subject = subjects[-1]
        train_val_subjects = subjects[:-1]
    val_count = max(1, int(len(train_val_subjects) * val_ratio)) if len(train_val_subjects) > 1 else 0
    val_subjects = train_val_subjects[-val_count:] if val_count > 0 else []
    train_subjects = train_val_subjects[:-val_count] if val_count > 0 else train_val_subjects
    splits = {"train": train_subjects, "val": val_subjects, "test": [test_subject]}
    return splits, test_subject


def unique_subject_ids(subject_ids: Iterable[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for subject_id in subject_ids:
        if subject_id in seen:
            continue
        seen.add(subject_id)
        unique.append(subject_id)
    return unique


def dump_splits(splits: Dict[str, List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))

