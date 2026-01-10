"""Data ingress utilities.

The ingest layer reads precomputed NPZ files (fixtures or processed Sleep-EDF
subjects) into `SubjectRecord` objects used by downstream modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from pydantic import BaseModel, field_validator


@dataclass(slots=True)
class SubjectRecord:
    subject_id: str
    signals: np.ndarray  # shape: (epochs, channels, samples)
    labels: np.ndarray  # shape: (epochs,)
    sample_rate: int
    session_id: str | None = None


class DataConfig(BaseModel):
    data_dir: Path
    stage_labels: List[str] = ["W", "N1", "N2", "N3", "REM"]
    channel: str = "Fpz-Cz"
    epoch_length_sec: int = 30
    sample_rate: int = 100

    @field_validator("data_dir")
    @classmethod
    def validate_dir(cls, value: Path) -> Path:
        if not value.exists():
            raise FileNotFoundError(
                f"DATA_DIR '{value}' missing. Set DATA_DIR env var or run scripts/make_fixture_subset.py"
            )
        return value


def load_subject_npz(path: Path) -> SubjectRecord:
    data = np.load(path, allow_pickle=False)
    subject_id = path.stem
    signals = data["signals"]
    labels = data["labels"].astype(str)
    sample_rate = int(data["sample_rate"])
    return SubjectRecord(
        subject_id=subject_id,
        session_id=subject_id,
        signals=signals,
        labels=labels,
        sample_rate=sample_rate,
    )


def load_subjects(config: DataConfig) -> Iterable[SubjectRecord]:
    npz_files = sorted(config.data_dir.glob("*.npz"))
    if npz_files:
        for npz_file in npz_files:
            yield load_subject_npz(npz_file)
        return
    from .edf import load_edf_subjects

    yield from load_edf_subjects(config)


def list_subject_ids(data_dir: Path) -> List[str]:
    npz_files = sorted(data_dir.glob("*.npz"))
    if npz_files:
        return [path.stem for path in npz_files]
    from .raw import find_edf_pairs, resolve_sleep_edfx_dir

    telemetry_dir = resolve_sleep_edfx_dir(data_dir)
    pairs = find_edf_pairs(telemetry_dir)
    return [subject_id for _psg, _hyp, subject_id, _night in pairs]


CANONICAL_STAGES = ("W", "N1", "N2", "N3", "REM")


def stage_to_index_map(stage_labels: Iterable[str]) -> dict[str, int]:
    labels = list(stage_labels)
    if len(set(labels)) != len(labels):
        raise ValueError("Stage labels must be unique")
    if set(labels) != set(CANONICAL_STAGES):
        missing = sorted(set(CANONICAL_STAGES) - set(labels))
        extra = sorted(set(labels) - set(CANONICAL_STAGES))
        raise ValueError(
            f"Stage labels must match {CANONICAL_STAGES}. Missing: {missing}. Extra: {extra}."
        )
    return {stage: idx for idx, stage in enumerate(labels)}
