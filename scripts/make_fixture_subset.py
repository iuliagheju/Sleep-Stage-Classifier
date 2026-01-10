"""Builds a tiny synthetic fixture subset for smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

from sleep_stager.data.fixtures import (
    DEFAULT_EPOCHS_PER_SUBJECT,
    DEFAULT_NUM_SUBJECTS,
    DEFAULT_NUM_PAIRS,
    build_edf_fixture_subset,
    build_fixture_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic smoke-test fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fixtures"),
        help="Destination directory for fixture files",
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=DEFAULT_NUM_SUBJECTS,
        help="Number of synthetic subjects",
    )
    parser.add_argument(
        "--epochs-per-subject",
        type=int,
        default=DEFAULT_EPOCHS_PER_SUBJECT,
        help="Number of epochs per subject",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fixture generation",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Optional Sleep-EDF root directory (uses EDF pairs when set)",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=DEFAULT_NUM_PAIRS,
        help="Number of EDF PSG/Hypnogram pairs to include when --raw-dir is set",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Reference EDF files in-place instead of copying into output dir",
    )
    args = parser.parse_args()
    if args.raw_dir is not None:
        build_edf_fixture_subset(
            args.output_dir,
            raw_dir=args.raw_dir,
            num_pairs=args.num_pairs,
            seed=args.seed,
            copy_files=not args.no_copy,
        )
        print(f"EDF fixture subset with {args.num_pairs} pairs written to {args.output_dir}")
    else:
        build_fixture_subset(
            args.output_dir,
            num_subjects=args.num_subjects,
            epochs_per_subject=args.epochs_per_subject,
            seed=args.seed,
        )
        print(f"Fixture subset with {args.num_subjects} subjects written to {args.output_dir}")


if __name__ == "__main__":
    main()
