"""Unified CLI entrypoint for sleep-stager commands."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import typer
import yaml
from omegaconf import OmegaConf

from .train import _hash_payload, _resolve_config, train as train_command
from ..data.fixtures import build_edf_fixture_subset, build_fixture_subset
from ..data.ingest import list_subject_ids
from ..data.raw import validate_sleep_edfx
from ..data.splits import unique_subject_ids
from ..eval.schema import validate_metrics_schema, validate_summary_schema
from ..eval.report import (
    aggregate_kfold_metrics,
    aggregate_loso_metrics,
    build_comparison_report,
    build_dataset_summary,
    write_comparison_report,
    write_dataset_summary,
)

ROOT = Path(__file__).resolve().parents[3]
app = typer.Typer(add_completion=False)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing spec file at {path}")
    return yaml.safe_load(path.read_text())


def _existing_loso_run(runs_root: Path, held_out_subject_id: str, config_hash: str) -> bool:
    for metrics_path in runs_root.glob("*/metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        if metrics.get("held_out_subject_id") != held_out_subject_id:
            continue
        if metrics.get("config_hash") != config_hash:
            continue
        return True
    return False


@app.command("train")
def train(
    config_path: str = typer.Option("configs", help="Directory containing Hydra configs"),
    config_name: str = typer.Option("default", help="Config file name inside config_path"),
    override: list[str] = typer.Option(
        [],
        "--override",
        help="Hydra-style override, e.g. --override model.family=classical",
    ),
) -> None:
    train_command(config_path=config_path, config_name=config_name, override=override)


@app.command("train-kfold")
def train_kfold(
    config_path: str = typer.Option("configs", help="Directory containing Hydra configs"),
    config_name: str = typer.Option("default", help="Config file name inside config_path"),
    override: list[str] = typer.Option(
        [],
        "--override",
        help="Hydra-style override, e.g. --override model.family=classical",
    ),
    runs_root: Path | None = typer.Option(
        None,
        "--runs-root",
        help="Optional output directory for k-fold run artifacts",
    ),
    max_folds: int | None = typer.Option(
        None,
        "--max-folds",
        min=1,
        help="Optional limit on the number of folds to run",
    ),
    aggregate: bool = typer.Option(
        True,
        "--aggregate/--no-aggregate",
        help="Write a k-fold summary after training",
    ),
) -> None:
    eval_spec = _load_yaml(ROOT / "specs" / "evaluation.yaml")
    secondary = eval_spec.get("secondary_protocol", {})
    protocol = str(secondary.get("name", "kfold_subject"))
    k = int(secondary.get("k", 5))
    if max_folds is not None and max_folds > k:
        raise ValueError(f"max_folds must be <= {k}")
    fold_count = max_folds or k
    output_root = runs_root or (Path("artifacts") / f"kfold_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    output_root.mkdir(parents=True, exist_ok=True)
    for fold_idx in range(fold_count):
        fold_overrides = list(override)
        fold_overrides.extend(
            [
                f"evaluation.protocol={protocol}",
                f"evaluation.fold_index={fold_idx}",
                f"artifacts.root={output_root}",
            ]
        )
        typer.echo(f"Running fold {fold_idx + 1}/{fold_count}")
        train_command(config_path=config_path, config_name=config_name, override=fold_overrides)
    if aggregate:
        metrics_paths = sorted(output_root.glob("*/metrics.json"))
        if not metrics_paths:
            typer.echo(f"No metrics.json files found under {output_root}", err=True)
            raise typer.Exit(code=1)
        runs = [json.loads(path.read_text()) for path in metrics_paths]
        summary = aggregate_kfold_metrics(runs)
        summary_path = output_root / "kfold_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        typer.echo(f"Wrote k-fold summary to {summary_path}")


@app.command("train-loso")
def train_loso(
    config_path: str = typer.Option("configs", help="Directory containing Hydra configs"),
    config_name: str = typer.Option("default", help="Config file name inside config_path"),
    override: list[str] = typer.Option(
        [],
        "--override",
        help="Hydra-style override, e.g. --override model.family=classical",
    ),
    runs_root: Path | None = typer.Option(
        None,
        "--runs-root",
        help="Optional output directory for LOSO run artifacts",
    ),
    max_subjects: int | None = typer.Option(
        None,
        "--max-subjects",
        min=1,
        help="Optional limit on the number of held-out subjects to run",
    ),
) -> None:
    base_cfg = _resolve_config(config_path, config_name, override)
    data_dir = Path(base_cfg.data.dir)
    subject_ids = sorted(unique_subject_ids(list_subject_ids(data_dir)))
    if not subject_ids:
        typer.echo(f"No subjects found under {data_dir}", err=True)
        raise typer.Exit(code=1)
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]
    output_root = runs_root or Path(getattr(base_cfg.artifacts, "root", "artifacts"))
    output_root.mkdir(parents=True, exist_ok=True)
    base_overrides = list(override)
    base_overrides.extend(
        [
            "evaluation.protocol=loso",
            f"artifacts.root={output_root}",
        ]
    )
    for subject_id in subject_ids:
        subject_overrides = list(base_overrides)
        subject_overrides.append(f"evaluation.held_out_subject_id={subject_id}")
        cfg = _resolve_config(config_path, config_name, subject_overrides)
        config_hash = _hash_payload(OmegaConf.to_container(cfg, resolve=True))
        if _existing_loso_run(output_root, subject_id, config_hash):
            typer.echo(f"Skipping {subject_id} (existing run detected).")
            continue
        typer.echo(f"Running LOSO with held-out subject {subject_id}")
        train_command(config_path=config_path, config_name=config_name, override=subject_overrides)


@app.command("make-fixture")
def make_fixture(
    out: Path = typer.Option(Path("data/fixtures/smoke"), "--out", help="Output fixture directory"),
    num_subjects: int = typer.Option(3, "--num-subjects", min=1, help="Number of subjects"),
    epochs_per_subject: int = typer.Option(20, "--epochs-per-subject", min=1, help="Epochs per subject"),
    seed: int = typer.Option(42, "--seed", help="Random seed for synthetic fixtures"),
    raw_dir: Path | None = typer.Option(
        None,
        "--raw-dir",
        help="Optional Sleep-EDF directory (uses real EDF pairs when set)",
    ),
    num_pairs: int = typer.Option(
        10,
        "--num-pairs",
        min=1,
        help="Number of EDF PSG/Hypnogram pairs to include when --raw-dir is set",
    ),
    copy_files: bool = typer.Option(
        True,
        "--copy/--no-copy",
        help="Copy EDF files into the fixture directory instead of referencing in-place",
    ),
) -> None:
    if raw_dir is not None:
        build_edf_fixture_subset(out, raw_dir=raw_dir, num_pairs=num_pairs, seed=seed, copy_files=copy_files)
        typer.echo(f"EDF fixture subset written to {out}")
    else:
        build_fixture_subset(out, num_subjects=num_subjects, epochs_per_subject=epochs_per_subject, seed=seed)
        typer.echo(f"Synthetic fixture subset written to {out}")


@app.command("fetch-data")
def fetch_data(
    dest: Path = typer.Option(
        Path("data/raw/sleep-edfx-1.0.0"),
        "--dest",
        help="Destination directory for sleep-edfx/1.0.0",
    ),
    download: bool = typer.Option(False, "--download", help="Attempt to run scripts/fetch_sleep_edf.sh"),
) -> None:
    if download:
        script = ROOT / "scripts" / "fetch_sleep_edf.sh"
        if not script.exists():
            raise typer.Exit(code=1)
        bash = shutil.which("bash")
        if not bash:
            typer.echo("bash not found; run scripts/fetch_sleep_edf.sh manually.", err=True)
            raise typer.Exit(code=1)
        env = os.environ.copy()
        env["DATA_DIR"] = str(dest)
        subprocess.run([bash, str(script)], check=True, cwd=str(ROOT), env=env)
    validate_sleep_edfx(dest)
    typer.echo(f"Validated Sleep-EDF data at {dest}")


@app.command("eval")
def eval_run(
    run: Path = typer.Option(..., "--run", help="Run directory with metrics.json"),
    out: Path | None = typer.Option(None, "--out", help="Optional output path for eval summary"),
) -> None:
    metrics_path = run / "metrics.json"
    if not metrics_path.exists():
        typer.echo(f"metrics.json not found at {metrics_path}", err=True)
        raise typer.Exit(code=1)
    metrics = json.loads(metrics_path.read_text())
    schema_path = ROOT / "specs" / "artifacts_schema.yaml"
    if schema_path.exists():
        import yaml

        schema = yaml.safe_load(schema_path.read_text())
        fields = schema.get("metrics_json", {}).get("fields", {})
        errors = validate_metrics_schema(metrics, fields)
        if errors:
            typer.echo("Schema validation failed:\n" + "\n".join(errors), err=True)
            raise typer.Exit(code=1)
    output_path = out or (run / "metrics_eval.json")
    payload = {
        "run_id": metrics.get("run_id"),
        "model_name": metrics.get("model_name"),
        "metrics": metrics.get("metrics", {}),
        "calibration": metrics.get("calibration", {}),
        "temporal": metrics.get("temporal", {}),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    typer.echo(f"Wrote evaluation summary to {output_path}")


@app.command("eval-folds")
def eval_folds(
    runs_root: Path = typer.Option(..., "--runs-root", help="Directory containing run subdirectories"),
    out: Path | None = typer.Option(None, "--out", help="Optional output path for k-fold summary"),
) -> None:
    metrics_paths = sorted(runs_root.glob("*/metrics.json"))
    if not metrics_paths:
        typer.echo(f"No metrics.json files found under {runs_root}", err=True)
        raise typer.Exit(code=1)
    runs = [json.loads(path.read_text()) for path in metrics_paths]
    summary = aggregate_kfold_metrics(runs)
    output_path = out or (runs_root / "kfold_summary.json")
    output_path.write_text(json.dumps(summary, indent=2))
    typer.echo(f"Wrote k-fold summary to {output_path}")


@app.command("eval-loso")
def eval_loso(
    runs_root: Path = typer.Option(..., "--runs-root", help="Directory containing LOSO run subdirectories"),
    out: Path | None = typer.Option(None, "--out", help="Optional output path for LOSO summary"),
) -> None:
    metrics_paths = sorted(runs_root.glob("*/metrics.json"))
    if not metrics_paths:
        typer.echo(f"No metrics.json files found under {runs_root}", err=True)
        raise typer.Exit(code=1)
    runs = [json.loads(path.read_text()) for path in metrics_paths]
    summary = aggregate_loso_metrics(runs)
    output_path = out or (runs_root / "loso_summary.json")
    output_path.write_text(json.dumps(summary, indent=2))
    typer.echo(f"Wrote LOSO summary to {output_path}")


@app.command("list-runs")
def list_runs(
    runs_root: Path = typer.Option(..., "--runs-root", help="Directory containing run subdirectories"),
) -> None:
    metrics_paths = sorted(runs_root.glob("*/metrics.json"))
    if not metrics_paths:
        typer.echo(f"No metrics.json files found under {runs_root}", err=True)
        raise typer.Exit(code=1)
    typer.echo("model_name\tprotocol\theld_out_subject_id\trun_id\tconfig_hash")
    for metrics_path in metrics_paths:
        metrics = json.loads(metrics_path.read_text())
        typer.echo(
            "\t".join(
                [
                    str(metrics.get("model_name", "")),
                    str(metrics.get("split_protocol", "")),
                    str(metrics.get("held_out_subject_id", "")),
                    str(metrics.get("run_id", "")),
                    str(metrics.get("config_hash", "")),
                ]
            )
        )


@app.command("summarize-runs")
def summarize_runs(
    runs_root: Path = typer.Option(..., "--runs-root", help="Directory containing run subdirectories"),
    out_dir: Path | None = typer.Option(None, "--out-dir", help="Optional output directory for summary files"),
) -> None:
    metrics_paths = sorted(runs_root.glob("*/metrics.json"))
    if not metrics_paths:
        typer.echo(f"No metrics.json files found under {runs_root}", err=True)
        raise typer.Exit(code=1)
    runs = [json.loads(path.read_text()) for path in metrics_paths]
    summary = build_dataset_summary(runs, dataset_root=str(runs_root))
    summary_root = out_dir or (runs_root / "_summary")
    summary_path = write_dataset_summary(summary, summary_root)
    schema_path = ROOT / "specs" / "summary_schema.yaml"
    if schema_path.exists():
        schema = yaml.safe_load(schema_path.read_text())
        fields = schema.get("summary_json", {}).get("fields", {})
        errors = validate_summary_schema(summary, fields)
        if errors:
            typer.echo("Summary schema validation failed:\n" + "\n".join(errors), err=True)
            raise typer.Exit(code=1)
    typer.echo(f"Wrote summary to {summary_path}")


@app.command("compare-runs")
def compare_runs(
    runs_root: Path = typer.Option(..., "--runs-root", help="Directory containing run subdirectories"),
    out_dir: Path | None = typer.Option(None, "--out-dir", help="Optional output directory for comparison files"),
) -> None:
    metrics_paths = sorted(runs_root.glob("*/metrics.json"))
    if not metrics_paths:
        typer.echo(f"No metrics.json files found under {runs_root}", err=True)
        raise typer.Exit(code=1)
    runs = [json.loads(path.read_text()) for path in metrics_paths]
    report = build_comparison_report(runs, dataset_root=str(runs_root))
    output_root = out_dir or (runs_root / "_comparison")
    report_path = write_comparison_report(report, output_root)
    typer.echo(f"Wrote comparison report to {report_path}")


if __name__ == "__main__":
    app()
