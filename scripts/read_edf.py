"""Quick utility to inspect and optionally plot EDF/EDF+ files."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect an EDF file and optionally plot it")
    parser.add_argument("edf_path", type=Path, help="Path to .edf/.EDF file")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload data into memory (needed for some operations)",
    )
    parser.add_argument(
        "--pick",
        default=None,
        help="Optional channel name to select (e.g., 'EEG Fpz-Cz')",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Open an interactive MNE plot window",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Seconds per page in plot mode (default: 30)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.edf_path.exists():
        raise SystemExit(f"File not found: {args.edf_path}")

    try:
        import mne
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Could not import mne. Install dependencies first.") from exc

    raw = mne.io.read_raw_edf(args.edf_path, preload=args.preload, verbose="ERROR")

    if args.pick:
        if args.pick not in raw.ch_names:
            available = ", ".join(raw.ch_names)
            raise SystemExit(f"Channel '{args.pick}' not found. Available: {available}")
        raw.pick([args.pick])

    print(f"File: {args.edf_path}")
    print(f"Sampling rate (Hz): {raw.info['sfreq']}")
    print(f"Channels ({len(raw.ch_names)}): {', '.join(raw.ch_names)}")
    print(f"Duration (s): {raw.times[-1]:.2f}")

    if raw.annotations is not None and len(raw.annotations) > 0:
        print(f"Annotations: {len(raw.annotations)}")
    else:
        print("Annotations: none")

    if args.plot:
        raw.plot(duration=args.duration, block=True)


if __name__ == "__main__":
    main()
