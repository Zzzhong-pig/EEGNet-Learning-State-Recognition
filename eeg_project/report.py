"""Generate deployment-ready evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path


def render_markdown_report(summary_path: str | Path, output_path: str | Path) -> str:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    lines = [
        "# EEG Model Evaluation Report",
        "",
        f"**Protocol:** {summary.get('protocol', 'n/a')}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean | Std |",
        "|--------|------|-----|",
    ]
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        if metric in summary:
            lines.append(
                f"| {metric} | {summary[metric]['mean']:.4f} | {summary[metric]['std']:.4f} |"
            )
    if "primary_macro_f1" in summary:
        lines.extend(
            [
                "",
                "## Primary Metrics (calibrated when enabled)",
                "",
                f"- macro-F1: **{summary['primary_macro_f1']['mean']:.4f}** ± {summary['primary_macro_f1']['std']:.4f}",
                f"- accuracy: **{summary['primary_accuracy']['mean']:.4f}** ± {summary['primary_accuracy']['std']:.4f}",
            ]
        )
    lines.extend(["", "## Per-fold", ""])
    for fold in summary.get("folds", []):
        lines.append(
            f"- Fold {fold['fold']}: acc={fold['accuracy']:.4f}, "
            f"bal_acc={fold['balanced_accuracy']:.4f}, macro_f1={fold['macro_f1']:.4f}"
        )
        if "calibrated_macro_f1" in fold:
            lines.append(f"  - calibrated macro_f1={fold['calibrated_macro_f1']:.4f}")
    content = "\n".join(lines) + "\n"
    Path(output_path).write_text(content, encoding="utf-8")
    return content
