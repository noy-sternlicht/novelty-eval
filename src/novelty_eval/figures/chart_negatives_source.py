#!/usr/bin/env python3
"""
Negatives-Source Figure
=======================
One text-width figure with two heatmaps side by side — pointwise on the left,
pairwise on the right. Rows are where the *negative* ideas came from (human, or
a generation backbone); columns are judge models; each cell is that judge's
absolute score, not a delta.

It answers a different question from `two_track_figure.ipynb`, which holds
the negatives fixed and varies the ablation. Here the ablation is fixed at the
baseline prompt and the negatives move, so the figure shows how far judge
quality travels as the thing being discriminated against gets stronger.

Reads the same `unified_<setup>[_filtered].json` files the merge writes, so no
merge re-run and no bootstrap. Needs one merge directory per track, named so the
row specs can refer to them:

    python chart_negatives_source.py \
        --track human=output/ablation_sweeps/merge_hvh_all_metrics \
        --track generated=output/ablation_sweeps/merge_vanilla_all_metrics \
        --out-dir output/paper_figures --formats pdf png

Options worth knowing:

    --metrics P:W         one figure per pair, pointwise metric then pairwise,
                          e.g. --metrics f1_macro:pairwise_accuracy
                               --metrics accuracy:pairwise_accuracy_no_ties
                          Repeatable; default f1_macro:pairwise_accuracy.
    --variant filtered    which accuracy report the numbers came from
                          (default: filtered).
    --center 50           colour is centred on chance and diverges from it, so
                          below-chance cells read red. --center none gives a
                          plain sequential ramp over the data range.
    --config FILE         YAML with rows, labels and titles, so the paper wording
                          lives outside argv. See configs/negatives_source_figure.yaml.

Every cell prints its own number, so the colour scale is redundant encoding and
is off by default, as is the footnote — a compact paper figure carries neither,
and the LaTeX caption states the metric and the scale direction instead. Both
come back with --colorbar and --footnote.

Two things the caption has to say, because the figure cannot: the human row comes
from a different merge, and therefore a different test set, than the generated
rows; and colour diverges around 50, so red is below chance.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from novelty_eval.figures import unified_chart_data
from novelty_eval.figures.common import (
    DELTA_COLORS,
    PDF_RCPARAMS,
    chart_data_path,
    load_config,
    save_figure,
)

SETUPS = ("pointwise", "pairwise")
DEFAULT_METRICS = {"pointwise": "f1_macro", "pairwise": "pairwise_accuracy"}

# Rows are (label, track, source). `baseline` pulls the track's un-ablated
# "Current" numbers — constant across that track's panels — and `ablation` pulls
# one named panel's post-ablation numbers. `{setup}` expands to pointwise or
# pairwise, since the backbone swaps are registered once per setup.
DEFAULT_ROWS = [
    {"label": "Human", "track": "human", "source": "baseline"},
    {"label": "sonnet-4-5", "track": "generated", "source": "baseline"},
    {"label": "opus-4-5", "track": "generated", "source": "ablation",
     "panel": "opus-4-5_backbone_{setup}_vanilla"},
    {"label": "gpt-5.1", "track": "generated", "source": "ablation",
     "panel": "gpt_5.1_backbone_{setup}_vanilla"},
    {"label": "gpt-5.4", "track": "generated", "source": "ablation",
     "panel": "gpt_5.4_backbone_{setup}_vanilla"},
]

DEFAULT_SETUP_LABELS = {"pointwise": "Pointwise", "pairwise": "Pairwise"}

# Vendor prefixes carry no information here — every column is a judge, and the
# family name alone identifies it. Dropping them buys horizontal space back from
# the rotated column labels.
STRIPPED_MODEL_PREFIXES = ("claude-",)


def _short_model(name: str) -> str:
    """Column label for a judge: the model name without its vendor prefix."""
    for prefix in STRIPPED_MODEL_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _parse_track(spec: str) -> tuple[str, Path]:
    """Parse a --track name=path/to/merge_dir argument."""
    if "=" not in spec:
        raise SystemExit(f"--track expects name=path, got: {spec!r}")
    name, raw = spec.split("=", 1)
    path = Path(raw.strip())
    if not path.exists():
        raise SystemExit(f"--track path does not exist: {path}")
    return name.strip(), path


def _parse_metric_pair(spec: str) -> dict[str, str]:
    """Parse a --metrics POINTWISE:PAIRWISE argument into a per-setup mapping."""
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) != 2 or not all(parts):
        raise SystemExit(
            f"--metrics expects POINTWISE_KEY:PAIRWISE_KEY, got: {spec!r}")
    return {"pointwise": parts[0], "pairwise": parts[1]}


def _data_path(merge_dir: Path, setup: str, variant: str) -> Path | None:
    """The saved chart data for one setup, or None when the merge has none."""
    path = chart_data_path(merge_dir, setup, variant)
    return path if path.exists() else None


def _baseline_metrics(data: dict, model: str) -> dict:
    """
    The track's un-ablated numbers for one judge.

    Every panel in a track carries the same baseline, so the first panel that has
    this judge answers for all of them.
    """
    for panel in data.get("panels", {}).values():
        metrics = (panel.get("baseline_metrics") or {}).get(model)
        if metrics:
            return metrics
    return {}


def _cell_value(row: dict, tracks: dict, setup: str, metric: str,
                model: str) -> float:
    """One cell: the score for this negatives source, judge and metric, or NaN."""
    data = tracks.get(row["track"])
    if data is None:
        return np.nan
    if row.get("source", "baseline") == "baseline":
        metrics = _baseline_metrics(data, model)
    else:
        name = str(row["panel"]).format(setup=setup)
        panel = data.get("panels", {}).get(name)
        if panel is None:
            return np.nan
        metrics = (panel.get("ablation_metrics") or {}).get(model) or {}
    value = metrics.get(metric)
    return np.nan if value is None else float(value) * 100.0


def _validate_rows(rows: list, tracks: dict, setup: str) -> None:
    """Warn once per row that resolves to nothing, rather than drawing a blank."""
    for row in rows:
        if row["track"] not in tracks:
            print(f"warning: row {row['label']!r} wants track {row['track']!r}, "
                  f"which was not given with --track.", file=sys.stderr)
            continue
        if row.get("source", "baseline") == "baseline":
            continue
        name = str(row["panel"]).format(setup=setup)
        if name not in tracks[row["track"]].get("panels", {}):
            available = sorted(tracks[row["track"]].get("panels", {}))
            print(f"warning: {setup}: no panel {name!r} in track "
                  f"{row['track']!r} — available: {available}", file=sys.stderr)


def _render(blocks: list, row_labels: list[str], model_labels: list[str],
            out_path: Path, *, center: float | None, value_label: str,
            row_axis_label: str | None, col_axis_label: str | None,
            colorbar: bool, footnote: str, width_in: float, formats=None,
            dpi: int = 300) -> list[Path] | None:
    """
    Draw the side-by-side score heatmaps.

    Args:
        blocks: [(title, values), ...] left to right; values is (n_rows x n_models)
            in percent, NaN where a cell has no data. One block per setup.
        center: colour pivot, e.g. 50 for chance — the scale then diverges
            symmetrically around it so below-chance cells read red. None ramps
            sequentially over the observed range instead.
        value_label: colourbar label, e.g. "score (%)".

    Returns the paths written, or None if matplotlib is unavailable or nothing is
    finite.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return None

    if not blocks or not row_labels or not model_labels:
        return None

    finite = np.concatenate([v[np.isfinite(v)] for _, v in blocks])
    if finite.size == 0:
        return None

    if center is None:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmax - vmin < 1e-9:
            vmin, vmax = vmin - 1.0, vmax + 1.0
        # Sequential: the pale end of the diverging ramp upward, so a low score
        # is light rather than red — red would imply a sign the scale lacks.
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "score_seq", DELTA_COLORS[1:], N=256)
    else:
        # Symmetric around the pivot so equal distances either side get equal
        # colour weight, and the pivot itself lands on the neutral midpoint.
        spread = max(float(np.abs(finite - center).max()), 1.0)
        vmin, vmax = center - spread, center + spread
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "score_div", DELTA_COLORS, N=256)
    cmap.set_bad(color="#e2e8f0")

    n_r, n_c, n_b = len(row_labels), len(model_labels), len(blocks)

    # Explicit inch-based budget, as in the two-track heatmap: the axes are placed
    # absolutely so cell size does not drift with row count.
    row_label_in, gap_in = 1.42, 0.16
    ylab_in = 0.18 if row_axis_label else 0.0
    left_in = ylab_in + row_label_in
    cell_w = (width_in - left_in - gap_in * (n_b - 1) - 0.04) / (n_b * n_c)
    cell_h = 0.40
    grid_h_in = n_r * cell_h
    bot_pad = 0.04
    foot_in = (0.20 + 0.11 * footnote.count("\n")) if footnote else 0.0
    cbar_lab_in, cbar_in = (0.30, 0.10) if colorbar else (0.0, 0.0)
    mid_gap_in = 0.34 if colorbar else (0.16 if footnote else 0.0)
    xlab_in, title_in, top_pad, top_gap_in = 0.40, 0.25, 0.03, 0.10
    xtitle_in = 0.22 if col_axis_label else 0.0
    grid_bottom_in = (bot_pad + foot_in + cbar_lab_in + cbar_in + mid_gap_in
                      + xtitle_in + xlab_in)
    fig_h = grid_bottom_in + grid_h_in + top_gap_in + title_in + top_pad

    plt.rcParams.update({"font.family": "sans-serif", **PDF_RCPARAMS})
    fig = plt.figure(figsize=(width_in, fig_h), facecolor="white")
    left = left_in / width_in
    block_w = (n_c * cell_w) / width_in
    gap = gap_in / width_in
    bottom, height = grid_bottom_in / fig_h, grid_h_in / fig_h

    block_axes = []
    for bi, (title, values) in enumerate(blocks):
        ax = fig.add_axes([left + bi * (block_w + gap), bottom, block_w, height])
        block_axes.append(ax)
        # pcolormesh rather than imshow: one vector rectangle per cell, so the PDF
        # stays sharp instead of embedding a tiny bitmap.
        ax.pcolormesh(np.arange(n_c + 1) - 0.5, np.arange(n_r + 1) - 0.5,
                      np.ma.masked_invalid(values),
                      cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
        ax.set_xlim(-0.5, n_c - 0.5)
        ax.set_ylim(n_r - 0.5, -0.5)   # row 0 at the top

        for i in range(n_r):
            for j in range(n_c):
                v = values[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=6, color="#999999")
                    continue
                norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                tc = "white" if norm_v < 0.25 or norm_v > 0.82 else "#1a1a1a"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.8, color=tc, fontweight="bold")

        for x in range(n_c + 1):
            ax.axvline(x - 0.5, color="white", linewidth=1.0)
        for y in range(n_r + 1):
            ax.axhline(y - 0.5, color="white", linewidth=1.0)
        ax.set_yticks(range(n_r))
        if bi == 0:
            ax.set_yticklabels(row_labels, fontsize=6.8)
            if row_axis_label:
                # set_ylabel measures the tick labels, so the title hugs the
                # longest row name instead of a guessed gutter width.
                ax.set_ylabel(row_axis_label, fontsize=7.4, fontweight="bold",
                              color="#333333", labelpad=7)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, pad=1.5)

        # Judge names sit under the grid; the block title owns the top.
        ax.set_xticks(range(n_c))
        ax.set_xticklabels(model_labels, fontsize=6.0, rotation=38, ha="right",
                           rotation_mode="anchor")
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(axis="x", length=0, pad=2.0)
        ax.text(0.5, 1.0 + top_gap_in / grid_h_in, title, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.30", fc="#eef1f5", ec="none"))

    blocks_w = block_w * n_b + gap * (n_b - 1)
    if col_axis_label:
        # Spans the blocks and hangs off the measured bottom of the rotated judge
        # names — those vary in length, so a fixed offset would leave a hole.
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        ticks = [t for a in block_axes for t in a.get_xticklabels()]
        y0 = min(t.get_window_extent(rend).y0 for t in ticks) / (fig_h * fig.dpi)
        fig.text(left + blocks_w / 2, y0 - 0.07 / fig_h, col_axis_label,
                 ha="center", va="top", fontsize=7.4, fontweight="bold",
                 color="#333333")

    if colorbar:
        cax = fig.add_axes([left, (bot_pad + foot_in + cbar_lab_in) / fig_h,
                            blocks_w, cbar_in / fig_h])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap),
            cax=cax, orientation="horizontal")
        cb.set_label(value_label, fontsize=6.6, labelpad=1.5)
        cb.ax.tick_params(labelsize=5.8, length=2, pad=1)
        cb.outline.set_visible(False)
        if center is not None:
            # The pivot is the whole point of the diverging scale; mark it.
            cb.ax.axvline(center, color="#333333", linewidth=0.8)
        # matplotlib rasterizes colourbar solids by default; keep the PDF vector.
        if cb.solids is not None:
            cb.solids.set_rasterized(False)

    if footnote:
        fig.text(left, bot_pad / fig_h, footnote, ha="left", va="bottom",
                 fontsize=5.4, color="#666666", style="italic", linespacing=1.5)

    return save_figure(fig, out_path, formats=formats, dpi=dpi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side pointwise/pairwise heatmaps over negatives sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--track", action="append", required=True, metavar="NAME=DIR",
                        help="Merge output directory, named for the row specs to "
                             "refer to (default rows use `human` and `generated`).")
    parser.add_argument("--out-dir", required=True, help="Where the figures go.")
    parser.add_argument("--metrics", action="append", metavar="POINTWISE:PAIRWISE",
                        help="Metric pair for one figure; repeat for more "
                             "(default: f1_macro:pairwise_accuracy).")
    parser.add_argument("--variant", default="filtered",
                        choices=["filtered", "unfiltered"],
                        help="Which accuracy report the data came from (default: filtered).")
    parser.add_argument("--models", nargs="+", metavar="NAME",
                        help="Judge models, left-to-right (default: all shared).")
    parser.add_argument("--center", default="50",
                        help="Colour pivot in percent, e.g. 50 for chance, or "
                             "`none` for a sequential scale (default: 50).")
    parser.add_argument("--config", help="YAML with rows / labels / titles / footnote.")
    parser.add_argument("--formats", nargs="+", default=["pdf"], metavar="EXT",
                        help="Output formats (default: pdf). PDF is vector.")
    parser.add_argument("--width", type=float, default=7.1,
                        help="Figure width in inches (default: 7.1, ~full text width).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Raster resolution; ignored for PDF.")
    parser.add_argument("--colorbar", action="store_true",
                        help="Draw the shared colour scale under the blocks. Off "
                             "by default: every cell prints its own number, so the "
                             "scale is redundant encoding on a compact figure.")
    parser.add_argument("--footnote", metavar="TEXT",
                        help="Text under the figure. Off by default; the caption "
                             "is where the caveats belong.")
    parser.add_argument("--row-axis-label", help="Name for what the rows are.")
    parser.add_argument("--col-axis-label", help="Name for what the columns are.")
    parser.add_argument("--prefix", default="negatives_source",
                        help="Output filename prefix (default: negatives_source).")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    rows_cfg = cfg.get("rows") or DEFAULT_ROWS
    if not isinstance(rows_cfg, list) or not rows_cfg:
        raise SystemExit("config `rows:` must be a non-empty list of row specs.")
    for row in rows_cfg:
        if not isinstance(row, dict) or "label" not in row or "track" not in row:
            raise SystemExit(f"row spec needs `label` and `track`: {row!r}")
        if row.get("source", "baseline") == "ablation" and "panel" not in row:
            raise SystemExit(f"row {row['label']!r} has source: ablation but no `panel`.")

    metric_labels_cfg = cfg.get("metric_labels", {}) or {}
    setup_labels = {**DEFAULT_SETUP_LABELS, **(cfg.get("setup_labels", {}) or {})}
    model_labels_cfg = cfg.get("model_labels", {}) or {}
    cfg_models = cfg.get("models") or None
    if cfg_models and not isinstance(cfg_models, list):
        raise SystemExit("config `models:` must be a list of judge model names.")
    colorbar = args.colorbar or bool(cfg.get("colorbar", False))
    footnote = args.footnote if args.footnote is not None else cfg.get("footnote", "")
    row_axis_label = (args.row_axis_label or cfg.get("row_axis_label")
                      or "Negatives source")
    col_axis_label = (args.col_axis_label or cfg.get("col_axis_label")
                      or "Judge model")
    value_label = cfg.get("value_label", "score (%)")

    if str(args.center).strip().lower() in {"none", "off", ""}:
        center = None
    else:
        try:
            center = float(args.center)
        except ValueError:
            raise SystemExit(f"--center wants a number or `none`, got {args.center!r}")

    track_dirs = dict(_parse_track(t) for t in args.track)
    metric_pairs = [_parse_metric_pair(m) for m in (args.metrics or [])] \
        or [dict(DEFAULT_METRICS)]

    # Load once per (track, setup); every metric pair re-reads the same panels.
    loaded: dict[str, dict[str, dict]] = {}
    for setup in SETUPS:
        loaded[setup] = {}
        for name, merge_dir in track_dirs.items():
            path = _data_path(merge_dir, setup, args.variant)
            if path is None:
                print(f"warning: no {args.variant} {setup} data for track "
                      f"{name!r} under {merge_dir}", file=sys.stderr)
                continue
            loaded[setup][name] = unified_chart_data.load(path)
        _validate_rows(rows_cfg, loaded[setup], setup)

    models = args.models or cfg_models
    if not models:
        per_track = [d["all_models"] for setup in SETUPS
                     for d in loaded[setup].values()]
        if not per_track:
            raise SystemExit("No chart data loaded — check --track paths and --variant.")
        shared = set(per_track[0])
        for m in per_track[1:]:
            shared &= set(m)
        models = [m for m in per_track[0] if m in shared]
    if not models:
        raise SystemExit("Tracks share no judge models.")

    out_dir = Path(args.out_dir)
    row_labels = [r["label"] for r in rows_cfg]
    written_any = False

    for metrics in metric_pairs:
        blocks = []
        for setup in SETUPS:
            if not loaded[setup]:
                continue
            metric = metrics[setup]
            # `available_metrics` is the merge's own name for the metric, so an
            # unknown key is a typo rather than a missing run — say which.
            known = next(iter(loaded[setup].values())).get("available_metrics", {})
            if known and metric not in known:
                raise SystemExit(
                    f"unknown {setup} metric {metric!r}. Available: "
                    f"{', '.join(sorted(known))}")
            values = np.array([
                [_cell_value(row, loaded[setup], setup, metric, model)
                 for model in models]
                for row in rows_cfg
            ], dtype=float)
            label = metric_labels_cfg.get(metric) or known.get(metric, metric)
            blocks.append((f"{setup_labels.get(setup, setup)} — {label}", values))

        if not blocks:
            print("skipping: no setup had data.", file=sys.stderr)
            continue

        stem = (f"{args.prefix}_{args.variant}_"
                f"{metrics['pointwise']}__{metrics['pairwise']}")
        out_path = out_dir / f"{stem}.{args.formats[0].lstrip('.')}"
        result = _render(
            blocks, row_labels,
            [model_labels_cfg.get(m) or _short_model(m) for m in models],
            out_path, center=center, value_label=value_label,
            row_axis_label=row_axis_label, col_axis_label=col_axis_label,
            colorbar=colorbar, footnote=footnote, width_in=args.width,
            formats=args.formats[1:], dpi=args.dpi,
        )
        if not result:
            print(f"could not render {metrics['pointwise']}/{metrics['pairwise']}: "
                  "no finite data — check the metric keys against the merge data.",
                  file=sys.stderr)
            continue
        written_any = True
        for p in result:
            print(f"{metrics['pointwise']} + {metrics['pairwise']} → {p}")

    if not written_any:
        raise SystemExit("Nothing was rendered.")


if __name__ == "__main__":
    main()
