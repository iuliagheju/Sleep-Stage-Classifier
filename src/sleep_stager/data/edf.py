"""Sleep-EDF EDF/Hypnogram loader using MNE."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .ingest import DataConfig, SubjectRecord
from .raw import _record_id_from_name, find_edf_pairs, resolve_sleep_edfx_dir


@dataclass(slots=True)
class AnnotationMapping:
    mapping: Dict[str, str]
    exclude: Tuple[str, ...] = ("M", "?", "UNKNOWN")


DEFAULT_MAPPING = AnnotationMapping(
    mapping={
        "W": "W",
        "1": "N1",
        "2": "N2",
        "3": "N3",
        "4": "N3",
        "R": "REM",
        "Sleep stage W": "W",
        "Sleep stage 1": "N1",
        "Sleep stage 2": "N2",
        "Sleep stage 3": "N3",
        "Sleep stage 4": "N3",
        "Sleep stage R": "REM",
    }
)


def _import_mne():
    try:
        import mne  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires optional dependency
        raise ImportError("Install mne to load EDF files: pip install mne") from exc
    return mne


def _labels_from_annotations(
    annotations, total_duration: float, epoch_length_sec: int, mapping: AnnotationMapping
) -> List[str]:
    n_epochs = int(total_duration // epoch_length_sec)
    labels: List[str] = [""] * n_epochs
    for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
        stage_key = desc.strip()
        if stage_key in mapping.exclude:
            continue
        if stage_key not in mapping.mapping:
            continue
        label = mapping.mapping[stage_key]
        start_epoch = int(onset // epoch_length_sec)
        end_epoch = int((onset + duration) // epoch_length_sec)
        end_epoch = min(end_epoch, n_epochs)
        for idx in range(start_epoch, end_epoch):
            labels[idx] = label
    return labels


def _valid_label_mask(labels: List[str]) -> np.ndarray:
    return np.array([bool(label) for label in labels])


def _segment_epochs(data: np.ndarray, sample_rate: int, epoch_length_sec: int) -> np.ndarray:
    epoch_samples = int(sample_rate * epoch_length_sec)
    total_samples = data.shape[1] - (data.shape[1] % epoch_samples)
    trimmed = data[:, :total_samples]
    epochs = trimmed.reshape(trimmed.shape[0], -1, epoch_samples).transpose(1, 0, 2)
    return epochs


def _match_channel(ch_names: List[str], desired: str) -> str | None:
    if desired in ch_names:
        return desired
    desired_lower = desired.lower()
    for name in ch_names:
        if name.lower() == desired_lower:
            return name
    for name in ch_names:
        lowered = name.lower()
        if lowered.endswith(desired_lower) or desired_lower in lowered:
            return name
    return None


def load_edf_subjects(config: DataConfig) -> Iterable[SubjectRecord]:
    mne = _import_mne()
    telemetry_dir = resolve_sleep_edfx_dir(config.data_dir)
    pairs = find_edf_pairs(telemetry_dir)
    if not pairs:
        raise FileNotFoundError(f"No EDF pairs found under {telemetry_dir}")
    for psg_path, hyp_path, subject_id, _night in pairs:
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
        selected_channel = _match_channel(raw.ch_names, config.channel)
        if not selected_channel:
            raise ValueError(f"Channel '{config.channel}' not found in {psg_path.name}")
        if int(raw.info["sfreq"]) != config.sample_rate:
            raise ValueError(f"Expected {config.sample_rate} Hz but found {raw.info['sfreq']} Hz")
        data = raw.get_data(picks=[selected_channel])
        annotations = mne.read_annotations(hyp_path)
        labels = _labels_from_annotations(
            annotations, total_duration=raw.times[-1], epoch_length_sec=config.epoch_length_sec, mapping=DEFAULT_MAPPING
        )
        epochs = _segment_epochs(data, config.sample_rate, config.epoch_length_sec)
        mask = _valid_label_mask(labels)
        n = min(mask.shape[0], epochs.shape[0])
        if n == 0:
            continue
        mask = mask[:n]
        epochs = epochs[:n][mask]
        labels = np.array(labels[:n])[mask].tolist()
        if not labels:
            continue
        record_id = _record_id_from_name(psg_path.stem)
        yield SubjectRecord(
            subject_id=subject_id,
            session_id=record_id,
            signals=epochs,
            labels=np.array(labels),
            sample_rate=config.sample_rate,
        )
