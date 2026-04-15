"""Training progress helpers for elapsed-time and ETA reporting."""
from __future__ import annotations

import math
import time


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class TrainingProgressTracker:
    def __init__(self, total_steps: int, label: str) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        self.total_steps = int(total_steps)
        self.label = label
        self._start_time = time.perf_counter()

    def message(self, step: int, detail: str = "") -> str:
        step = max(0, min(int(step), self.total_steps))
        elapsed = time.perf_counter() - self._start_time
        eta = math.inf if step == 0 else (elapsed / step) * (self.total_steps - step)
        eta_text = "--:--" if not math.isfinite(eta) else format_duration(eta)
        suffix = f" {detail}" if detail else ""
        return (
            f"[train/{self.label}] step={step}/{self.total_steps} "
            f"elapsed={format_duration(elapsed)} eta={eta_text}{suffix}"
        )
