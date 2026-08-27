#!/usr/bin/env python3
"""Aggregate measured run records and emit traceable tables and SVG plots."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = {"bace": "roc_auc", "tox21": "macro_roc_auc", "lipophilicity": "rmse"}


def _read_run(path: Path, category: str) -> dict[str, object] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "final" in data:
        metrics = data["final"]["test"]["metrics"]
        dataset = data["dataset"]
        split = data["split"]["method"]
        model = path.parents[1].name
        parameters = data["parameter_counts"]["total"]
        trainable_parameters = data["parameter_counts"]["trainable"]
    elif "test" in data:
        metrics = data["test"]
        dataset, split, model = data["dataset"], data["split"], data["model"]
        parameters = trainable_parameters = data["parameter_count"]
    else:
        return None
    metric = PRIMARY[dataset]
    value = metrics.get(metric)
    if value is None:
        return None
    return {
        "category": category, "dataset": dataset, "split": split, "model": model,
        "seed": int(data["seed"]), "metric": metric, "value": float(value),
        "parameters": int(parameters), "trainable_parameters": int(trainable_parameters),
        "metrics_path": path,
    }


def collect(root: Path, category: str) -> list[dict[str, object]]:
    records = []
    if root.exists():
        for path in sorted(root.glob("**/metrics.json")):
            record = _read_run(path, category)
            if record is not None:
                records.append(record)
    return records


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["split"], record["model"], record["metric"])].append(record)
    rows = []
    for (dataset, split, model, metric), values in sorted(groups.items()):
        scores = np.asarray([value["value"] for value in values], dtype=np.float64)
        rows.append({
            "dataset": dataset, "split": split, "model": model, "metric": metric,
            "mean": float(scores.mean()), "standard_deviation": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "valid_runs": len(scores), "seeds": " ".join(str(value["seed"]) for value in values),
            "mean_parameter_count": round(float(np.mean([value["parameters"] for value in values]))),
            "mean_trainable_parameter_count": round(
                float(np.mean([value["trainable_parameters"] for value in values]))
            ),
            "result_scope": "project-measured compact empirical reproduction",
        })
    return rows


def _write_csv(path: Path, rows) -> None:
    fields = ["dataset", "split", "model", "metric", "mean", "standard_deviation",
              "valid_runs", "seeds", "mean_parameter_count", "mean_trainable_parameter_count",
              "result_scope"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows, title):
    lines = [f"## {title}", "", "| Dataset | Split | Model | Metric | Mean ± SD | n | Total / trainable parameters |",
             "|---|---|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['model']} | {row['metric']} | "
            f"{row['mean']:.4f} ± {row['standard_deviation']:.4f} | {row['valid_runs']} | "
            f"{row['mean_parameter_count']} / {row['mean_trainable_parameter_count']} |"
        )
    return "\n".join(lines)


def _update_readme(reproduction_rows, ablation_rows) -> None:
    path = PROJECT_ROOT / "README.md"
    if not path.is_file():
        return
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    text = path.read_text(encoding="utf-8")
    if begin not in text or end not in text:
        return
    full_rows = [row for row in reproduction_rows if row["model"] == "full_gtpro"]
    generated = (
        begin + "\n\n" + _markdown(full_rows, "Compact GTpro reproduction") + "\n\n"
        + _markdown(ablation_rows, "BACE one-seed ablation screening") + "\n\n" + end
    )
    prefix, remainder = text.split(begin, 1)
    _, suffix = remainder.split(end, 1)
    path.write_text(prefix + generated + suffix, encoding="utf-8")


def _svg_bars(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    width, height = 900, max(260, 90 + 36 * len(labels))
    finite = [abs(value) for value in values if np.isfinite(value)] or [1.0]
    scale = 600 / max(finite)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="32" font-family="sans-serif" font-size="20" font-weight="bold">{title}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 62 + index * 36
        bar_width = max(1, abs(value) * scale)
        lines.append(f'<text x="20" y="{y + 16}" font-family="sans-serif" font-size="13">{label}</text>')
        lines.append(f'<rect x="250" y="{y}" width="{bar_width:.1f}" height="22" fill="#3973ac"/>')
        lines.append(f'<text x="{260 + bar_width:.1f}" y="{y + 16}" font-family="monospace" font-size="13">{value:.4f}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_trace(records, output: Path) -> None:
    """Copy lightweight evidence while removing machine-specific root paths."""

    trace_root = output / "run_records"
    if trace_root.exists():
        shutil.rmtree(trace_root)
    for record in records:
        source_dir = Path(record["metrics_path"]).parent
        relative = Path(record["category"]) / source_dir.relative_to(PROJECT_ROOT / "runs" / record["category"])
        destination = trace_root / relative
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "config.yaml", "environment.json"):
            source = source_dir / name
            if source.is_file():
                content = source.read_text(encoding="utf-8")
                content = content.replace(str(PROJECT_ROOT), "${PROJECT_ROOT}")
                (destination / name).write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GTpro reproduction and ablation runs")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reproduction_records = collect(args.runs_root / "reproduction", "reproduction")
    ablation_records = collect(args.runs_root / "ablation/bace_random", "ablation")
    reproduction_rows, ablation_rows = aggregate(reproduction_records), aggregate(ablation_records)
    _write_csv(args.output_dir / "reproduction.csv", reproduction_rows)
    _write_csv(args.output_dir / "ablation.csv", ablation_rows)
    text = (
        "# Generated empirical result tables\n\n"
        "All values below are project-measured compact-model results generated from copied `metrics.json` "
        "records. They are not values reported by the original paper.\n\n"
        + _markdown(reproduction_rows, "Reproduction and baselines") + "\n\n"
        + _markdown(ablation_rows, "One-seed ablation screening") + "\n"
    )
    (args.output_dir / "README_tables.md").write_text(text, encoding="utf-8")
    full_rows = [row for row in reproduction_rows if row["model"] == "full_gtpro"]
    _svg_bars(
        args.output_dir / "random_scaffold_comparison.svg", "Compact GTpro: random vs scaffold",
        [f"{row['dataset']} / {row['split']} / {row['metric']}" for row in full_rows],
        [row["mean"] for row in full_rows],
    )
    _svg_bars(
        args.output_dir / "ablation_bace.svg", "BACE random one-seed ablation ROC-AUC",
        [row["model"] for row in ablation_rows], [row["mean"] for row in ablation_rows],
    )
    _copy_trace(reproduction_records + ablation_records, args.output_dir)
    _update_readme(reproduction_rows, ablation_rows)
    print(f"Wrote {len(reproduction_rows)} reproduction rows and {len(ablation_rows)} ablation rows")


if __name__ == "__main__":
    main()
