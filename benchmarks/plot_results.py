#!/usr/bin/env python
"""Render benchmark results into charts for the README.

    python benchmarks/plot_results.py

Reads ``benchmarks/results/*_latest.json`` (produced by ``kubeai bench``) and
writes PNGs next to them. Deliberately dependency-light: uses matplotlib when it
is installed and otherwise emits an SVG by hand, so the charts can always be
regenerated in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}_latest.json"
    if not path.exists():
        print(f"skip: {path} not found (run `make bench` first)", file=sys.stderr)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _svg_bar_chart(
    path: Path, title: str, labels: list[str], values: list[float], ylabel: str
) -> None:
    """A minimal hand-rolled bar chart, so this works with zero dependencies."""
    width, height = 720, 400
    margin_left, margin_bottom, margin_top = 70, 90, 50
    plot_w = width - margin_left - 30
    plot_h = height - margin_bottom - margin_top
    peak = max(values) if values else 1.0
    bar_w = plot_w / max(len(values), 1) * 0.6
    gap = plot_w / max(len(values), 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="16" '
        f'font-weight="600">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - 30}" '
        f'y2="{margin_top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="14" y="{margin_top + plot_h / 2}" font-size="12" fill="#555" '
        f'transform="rotate(-90 14 {margin_top + plot_h / 2})" text-anchor="middle">{ylabel}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        bar_h = (value / peak) * plot_h if peak else 0
        x = margin_left + index * gap + (gap - bar_w) / 2
        y = margin_top + plot_h - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
            f'fill="#2563eb" rx="3"/>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#111">{value:,.0f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{margin_top + plot_h + 18}" '
            f'text-anchor="middle" font-size="11" fill="#444">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {path}")


def _try_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def plot_serving(report: dict) -> None:
    scenarios = report["scenarios"]
    labels = [s["label"].replace(" batching", "\nbatching").replace(" (", "\n(") for s in scenarios]
    throughput = [s["throughput_tokens_per_second"] for s in scenarios]
    ttft = [s["mean_ttft_seconds"] * 1000 for s in scenarios]
    tpot = [s["mean_tpot_seconds"] * 1000 for s in scenarios]

    if _try_matplotlib():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, values, title, ylabel, colour in zip(
            axes,
            [throughput, ttft, tpot],
            ["Throughput", "Time to first token", "Inter-token latency"],
            ["tokens / second", "milliseconds", "milliseconds"],
            ["#2563eb", "#f59e0b", "#10b981"],
            strict=True,
        ):
            bars = ax.bar(range(len(values)), values, color=colour)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.bar_label(bars, fmt="%.1f", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(
            f"kubeai serving — {report['config']['requests']} requests, "
            f"{report['config']['max_tokens']} max tokens, backend={report['config']['backend']}"
        )
        fig.tight_layout()
        out = RESULTS / "serving_benchmark.png"
        fig.savefig(out, dpi=140)
        print(f"wrote {out}")
        return

    _svg_bar_chart(RESULTS / "serving_throughput.svg", "Serving throughput", labels,
                   throughput, "tokens / second")
    _svg_bar_chart(RESULTS / "serving_ttft.svg", "Time to first token", labels,
                   ttft, "milliseconds")


def plot_workflow(report: dict) -> None:
    scenarios = report["scenarios"]
    labels = [s["label"] for s in scenarios]
    durations = [s["duration_seconds"] for s in scenarios]

    if _try_matplotlib():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.bar(labels, durations, color="#7c3aed")
        ax.set_ylabel("wall-clock seconds")
        ax.set_title(
            f"Workflow engine — {report['config']['tasks']} tasks, "
            f"{report['config']['seconds_per_step']}s per step"
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out = RESULTS / "workflow_benchmark.png"
        fig.savefig(out, dpi=140)
        print(f"wrote {out}")
        return

    _svg_bar_chart(RESULTS / "workflow_duration.svg", "Workflow wall-clock time",
                   labels, durations, "seconds")


def main() -> int:
    serving = load("bench_serving")
    if serving:
        plot_serving(serving)
    workflow = load("bench_workflow")
    if workflow:
        plot_workflow(workflow)
    if not serving and not workflow:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
