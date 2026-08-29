#!/usr/bin/env python3
"""
Plot pointwise and pairwise evaluation results grouped by setting.

For each metric that appears in the data, one figure is produced with up to two
subfigures — left for pointwise, right for pairwise.  Within each subfigure the
X-axis shows settings and bars are grouped by model.

YAML input format
-----------------
pointwise:
  setting-1:
    gpt-5.1: {f1_macro: 0.78, f1_pos: 0.81, f1_neg: 0.74}
    gpt-5.2: {f1_macro: 0.80, f1_pos: 0.83, f1_neg: 0.76}
  setting-2:
    gpt-5.1: {f1_macro: 0.75}
    gpt-5.2: {f1_macro: 0.77}

pairwise:
  setting-1:
    gpt-5.1: {pairwise_accuracy: 0.65, pairwise_accuracy_strict: 0.60}
    gpt-5.2: {pairwise_accuracy: 0.70, pairwise_accuracy_strict: 0.65}

# Optional — colors assigned to models in the order they first appear.
color_palette: ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

# Optional — directory to save figures (defaults to current directory).
output_dir: ./charts
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

_DEFAULT_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]

_STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "xtick.bottom": False,
    "ytick.left": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def _fmt_metric(name: str) -> str:
    """'f1_macro' → 'F1 Macro', 'pairwise_accuracy_strict' → 'Accuracy Strict'."""
    words = name.split("_")
    if words[0].lower() == "pairwise":
        words = words[1:]
    return " ".join(
        w.upper() if w.lower() in {"f1", "pos", "neg"} else w.capitalize()
        for w in words
    )


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _collect_models_and_metrics(data: dict) -> tuple[list[str], dict[str, set[str]]]:
    """Return ordered model list and {eval_type: {metric, ...}} from raw YAML data."""
    models: list[str] = []
    seen_models: set[str] = set()
    metrics: dict[str, set[str]] = {"pointwise": set(), "pairwise": set()}

    for eval_type in ("pointwise", "pairwise"):
        for _setting, model_dict in (data.get(eval_type) or {}).items():
            for model, scores in (model_dict or {}).items():
                if model not in seen_models:
                    models.append(model)
                    seen_models.add(model)
                metrics[eval_type].update((scores or {}).keys())

    return models, metrics


def _grouped_bar(
    ax: plt.Axes,
    settings: list[str],
    models: list[str],
    values: dict[str, dict[str, float]],  # setting → model → value (or None)
    metric: str,
    colors: dict[str, str],
) -> None:
    n_settings = len(settings)
    n_models = len(models)
    group_gap = 0.25
    width = (1.0 - group_gap) / n_models
    x = np.arange(n_settings)

    all_vals: list[float] = []
    for i, model in enumerate(models):
        vals = [values.get(s, {}).get(model) for s in settings]
        numeric = [v if v is not None else 0.0 for v in vals]
        missing = [v is None for v in vals]
        offsets = x + (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            offsets, numeric, width=width * 0.92,
            color=colors[model], label=model,
            linewidth=0, zorder=3,
        )
        for bar, is_missing, val in zip(bars, missing, vals):
            if is_missing:
                bar.set_alpha(0.25)
                bar.set_hatch("//")
            elif val is not None:
                all_vals.append(val)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7, color="#444444",
                    zorder=4,
                )

    # Smart y-axis: zoom in on actual data range with padding
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        span = max(hi - lo, 0.02)
        ax.set_ylim(max(0, lo - span * 0.5), hi + span * 0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=0, ha="center", fontsize=9,
                       fontweight="bold")
    ax.set_ylabel(_fmt_metric(metric), fontweight="bold", fontsize=9, labelpad=8)
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))


def _build_values(
    data: dict, eval_type: str, settings: list[str], metric: str
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for setting in settings:
        model_dict = (data.get(eval_type) or {}).get(setting) or {}
        result[setting] = {
            model: scores[metric]
            for model, scores in model_dict.items()
            if isinstance(scores, dict) and metric in scores
        }
    return result


def plot(cfg_path: Path, output_dir: Path) -> None:
    data = _load(cfg_path)
    palette = data.get("color_palette") or _DEFAULT_PALETTE

    models, type_metrics = _collect_models_and_metrics(data)
    if not models:
        sys.exit("No model data found in the YAML file.")

    colors = {m: palette[i % len(palette)] for i, m in enumerate(models)}

    pointwise_settings: list[str] = list((data.get("pointwise") or {}).keys())
    pairwise_settings: list[str] = list((data.get("pairwise") or {}).keys())

    all_metrics: list[str] = list(
        OrderedDict.fromkeys(
            list(type_metrics["pointwise"]) + list(type_metrics["pairwise"])
        )
    )

    if not all_metrics:
        sys.exit("No metrics found in the YAML file.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in all_metrics:
        has_pw = metric in type_metrics["pointwise"] and bool(pointwise_settings)
        has_pa = metric in type_metrics["pairwise"] and bool(pairwise_settings)

        n_panels = int(has_pw) + int(has_pa)
        if n_panels == 0:
            continue

        n_settings = max(
            len(pointwise_settings) if has_pw else 0,
            len(pairwise_settings) if has_pa else 0,
        )
        panel_w = max(4.5, n_settings * len(models) * 0.55)
        with mpl.rc_context(_STYLE):
            fig, axes = plt.subplots(
                1, n_panels,
                figsize=(panel_w * n_panels, 4.2),
                constrained_layout=True,
            )
            if n_panels == 1:
                axes = [axes]

            panel = 0
            if has_pw:
                vals = _build_values(data, "pointwise", pointwise_settings, metric)
                _grouped_bar(
                    axes[panel], pointwise_settings, models, vals,
                    metric, colors,
                )
                panel += 1

            if has_pa:
                vals = _build_values(data, "pairwise", pairwise_settings, metric)
                _grouped_bar(
                    axes[panel], pairwise_settings, models, vals,
                    metric, colors,
                )

            handles = [
                mpl.patches.Patch(facecolor=colors[m], label=m) for m in models
            ]
            leg = fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(len(models), 6),
                bbox_to_anchor=(0.5, -0.14),
                frameon=True,
                fontsize=9,
                handlelength=1.2,
                handleheight=0.9,
                columnspacing=1.0,
            )
            leg.get_frame().set_facecolor("#f0f0f0")
            leg.get_frame().set_edgecolor("#cccccc")
            leg.get_frame().set_linewidth(0.8)

        safe_metric = metric.replace(" ", "_").replace("/", "-")
        out_path = output_dir / f"{safe_metric}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot pointwise/pairwise eval results grouped by setting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", type=Path, help="Path to the YAML config file.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for output figures (overrides config's output_dir).",
    )
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit(f"Config not found: {args.config}")

    raw = _load(args.config)
    output_dir = args.output_dir or Path(raw.get("output_dir", "."))
    plot(args.config, output_dir)


if __name__ == "__main__":
    main()
