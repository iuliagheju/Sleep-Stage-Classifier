"""Run training with live elapsed/ETA logs and a final wall-clock summary.

This launcher reuses the existing CLI training entrypoint. For deep models
(CNN/SEQ/Attention), per-epoch elapsed/ETA is printed from the model training
loops.
"""
from __future__ import annotations

import argparse
import time

from sleep_stager.cli.train import train as train_command
from sleep_stager.utils.progress import format_duration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sleep-stager training with ETA output")
    parser.add_argument(
        "--config-path",
        default="configs",
        help="Directory containing Hydra configs",
    )
    parser.add_argument(
        "--config-name",
        default="default",
        help="Config filename inside --config-path",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override (repeatable), e.g. --override model.family=cnn",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    print(
        f"[train-with-eta] starting config={args.config_name} overrides={len(args.override)}",
        flush=True,
    )
    train_command(
        config_path=str(args.config_path),
        config_name=str(args.config_name),
        override=list(args.override),
    )
    elapsed = time.perf_counter() - started
    print(f"[train-with-eta] total_elapsed={format_duration(elapsed)}", flush=True)


if __name__ == "__main__":
    main()
