#!/usr/bin/env python3
"""
Two-Track Ablation Figures
==========================
Build the compact, paper-ready ablation figures from the `unified_*.json` data
files that `merge_ablation_runs.py` writes — no merge re-run, no bootstrap.

Compared with the wide `unified_*.png` charts this transposes the layout
(rows = ablations, columns = judge models) and puts one column block per data
type side by side, so the figure grows *downward* as ablations are added rather
than sideways. Two setups → two figures, each one text-width.

Needs one merge output directory per track, e.g.:

    python chart_two_track.py \
        --track "Human-only"=output/ablation_sweeps/merge_hvh \
        --track "Human + generated"=output/ablation_sweeps/merge_vanilla \
        --out-dir output/paper_figures --formats pdf png

Options worth knowing:

    --stacked --compact   one-column layout: the track blocks stack vertically,
                          every gutter tightens, and the cells are fitted to the
                          numbers rather than stretched to fill the page — so
                          --width becomes a ceiling. --cell-width/--cell-height
                          override the fit.
    --style forest        dot-and-interval instead of the heatmap. Needs chart
                          data at schema v2+ (re-run the merge with
                          --charts-only if yours predates it).
    --style both          emit the heatmap and the forest plot.
    --variant filtered    which accuracy report to read (default: filtered).
    --rows a b c          canonical ablation names, in order. Also settable as
                          `rows:` in --config; the flag wins. Default: every
                          ablation the tracks have in common.
    --metric KEY          default: pairwise_accuracy / f1_macro per setup.
    --config FILE         YAML with row labels, track names and titles, so the
                          wording lives with the paper rather than in argv.

Row labels default to each ablation's title from the merge data. Those describe
the *change being made*, and some ablations add a capability rather than remove
one — check the wording reads correctly in the figure, or override it via
--config. See `two_track_figures.example.yaml`.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from novelty_eval.figures import unified_chart_data
from novelty_eval.figures.viz import (
    _generate_two_track_forest,
    _generate_two_track_heatmap,
)

DEFAULT_METRIC = {"pairwise": "pairwise_accuracy", "pointwise": "f1_macro"}
DEFAULT_METRIC_LABEL = {
    "pairwise": r"$\Delta$ pairwise accuracy (pts)",
    "pointwise": r"$\Delta$ macro-F1 (pts)",
}
SETUPS = ("pairwise", "pointwise")


def _parse_track(spec: str) -> tuple[str, Path]:
    """Parse a --track "Display name"=path/to/merge_dir argument."""
    if "=" not in spec:
        raise SystemExit(
            f'--track expects "Display name"=path, got: {spec!r}')
    title, raw = spec.split("=", 1)
    path = Path(raw.strip())
    if not path.exists():
        raise SystemExit(f"--track path does not exist: {path}")
    return title.strip(), path


def _data_path(merge_dir: Path, setup: str, variant: str) -> Path | None:
    """Locate unified_<setup>[_filtered].json under a merge output directory."""
    base = merge_dir / "unified_charts" if (merge_dir / "unified_charts").is_dir() \
        else merge_dir
    suffix = "" if variant == "unfiltered" else f"_{variant}"
    candidate = base / f"unified_{setup}{suffix}.json"
    return candidate if candidate.exists() else None


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError:
        raise SystemExit("--config needs PyYAML installed.")
    if not path.exists():
        example = Path(__file__).with_name("two_track_figures.example.yaml")
        hint = f"\nStart from the example:\n  cp {example} {path}" \
            if example.exists() else ""
        raise SystemExit(f"--config file not found: {path}{hint}")
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"--config is not valid YAML ({path}): {exc}")


# The same conceptual ablation is named per track *and* per setup when its
# instance set differs: convert_to_plan_form_pairwise_human in one place,
# convert_to_plan_form_pointwise_vanilla in another. Rows are matched on the name
# with these trailing tokens stripped, so a single `rows:` entry covers both
# setups — otherwise a setup-specific name would silently drop the row from the
# other setup's figure.
DEFAULT_TRACK_SUFFIXES = ("human", "vanilla", "hvh", "vanilla_ai", "ai",
                          "pairwise", "pointwise", "ranking")


# claude-sonnet-4-5 has no reasoning-effort control, so any number it produces
# under the reasoning ablation is an artefact: blank the cell instead.
UNSUPPORTED_CELLS = {("low_judge_reasoning", "claude-sonnet-4-5")}


def _drop_unsupported_cells(data: dict) -> dict:
    """Blank cells for judges that cannot honour the ablation being drawn."""
    for name, panel in data.get("panels", {}).items():
        key = _row_key(name, DEFAULT_TRACK_SUFFIXES)
        for ablation, model in UNSUPPORTED_CELLS:
            if key == ablation:
                panel.get("ablation_metrics", {}).pop(model, None)
    return data


def _row_key(name: str, suffixes: tuple[str, ...]) -> str:
    """
    Strip trailing track/setup tokens so per-track and per-setup variants of one
    ablation share a row key. Applied repeatedly: `..._pairwise_human` loses both.

    Never strips the whole name — a token that is the entire name is left alone.
    """
    ordered = sorted(suffixes, key=len, reverse=True)
    changed = True
    while changed:
        changed = False
        for suffix in ordered:
            token = "_" + suffix
            if name.endswith(token) and len(name) > len(token):
                name = name[: -len(token)]
                changed = True
                break
    return name


def _common_rows(track_data_list: list[dict], requested: list[str] | None,
                 labels: dict[str, str], aliases: dict[str, list[str]] | None = None,
                 suffixes: tuple[str, ...] = DEFAULT_TRACK_SUFFIXES) -> list:
    """
    Build the row spec: [(display_label, (canonical_per_track, ...)), ...].

    Each track is indexed by row key (the canonical name with a trailing track
    token removed), so `..._human` in one track lines up with `..._vanilla` in
    another. `aliases` maps a row key to extra acceptable names, for pairs the
    suffix rule cannot relate.

    A row that resolves in no track is dropped with a warning; one that resolves
    in some tracks renders as empty cells in the rest.
    """
    aliases = aliases or {}
    indexes = []
    for data in track_data_list:
        idx: dict[str, str] = {}
        for name in data.get("panels", {}):
            idx.setdefault(_row_key(name, suffixes), name)
            idx.setdefault(name, name)
        indexes.append(idx)

    def resolve(name: str, idx: dict[str, str]) -> str | None:
        key = _row_key(name, suffixes)
        if name in idx:
            return idx[name]
        if key in idx:
            return idx[key]
        for alt in aliases.get(key, []) + aliases.get(name, []):
            if alt in idx:
                return idx[alt]
        return None

    if requested:
        names = requested
    else:
        # Default to what the first track actually drew, keeping only rows that
        # resolve in every track — the union would drag in every non-rendered
        # ablation (backbone swaps and such) and defeat the point of a compact
        # figure.
        names = [c for c in track_data_list[0].get("rendered", [])
                 if all(resolve(c, idx) for idx in indexes)]

    rows = []
    for name in names:
        canonicals = tuple(resolve(name, idx) for idx in indexes)
        if not any(canonicals):
            print(f"warning: ablation {name!r} is in no track — skipping row.",
                  file=sys.stderr)
            continue
        label = labels.get(name) or labels.get(_row_key(name, suffixes))
        if label is None:
            # Use whichever track has a title for it.
            for data, canon in zip(track_data_list, canonicals):
                if canon and data["panels"][canon].get("title"):
                    label = data["panels"][canon]["title"]
                    break
        rows.append((label or name, canonicals))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build compact two-track ablation figures from saved chart data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--track", action="append", required=True, metavar='"Name"=DIR',
                        help="Repeat once per data type, in left-to-right order.")
    parser.add_argument("--out-dir", required=True, help="Where the figures go.")
    parser.add_argument("--setup", choices=SETUPS, action="append",
                        help="Limit to this setup (repeatable; default: both).")
    parser.add_argument("--variant", default="filtered",
                        choices=["filtered", "unfiltered"],
                        help="Which accuracy report the data came from (default: filtered).")
    parser.add_argument("--style", default="heatmap",
                        choices=["heatmap", "forest", "both"],
                        help="Figure style (default: heatmap).")
    parser.add_argument("--rows", nargs="+", metavar="NAME",
                        help="Canonical ablation names, in top-to-bottom order. "
                             "Overrides `rows:` in --config.")
    parser.add_argument("--models", nargs="+", metavar="NAME",
                        help="Judge models, in left-to-right order (default: all shared).")
    parser.add_argument("--metric", help="Metric key (default: per-setup default).")
    parser.add_argument("--metric-label", help="Axis/colourbar label for the metric.")
    parser.add_argument("--config", help="YAML with row_labels / track titles / metric labels.")
    parser.add_argument("--formats", nargs="+", default=["pdf"], metavar="EXT",
                        help="Output formats (default: pdf). PDF is vector.")
    parser.add_argument("--width", type=float, default=7.1,
                        help="Figure width in inches (default: 7.1, ~full text width).")
    parser.add_argument("--stacked", action="store_true",
                        help="Stack the track blocks vertically instead of side by "
                             "side, for a one-column figure. Pair with --width 3.3 "
                             "and --compact.")
    parser.add_argument("--compact", action="store_true",
                        help="Tighter cells, gutters and type, and flat judge names.")
    parser.add_argument("--model-label-rotation", type=float, metavar="DEG",
                        help="Angle for the judge names (default: 38, or 0 with "
                             "--compact).")
    parser.add_argument("--cell-width", type=float, metavar="IN",
                        help="Fix the cell width in inches (default: the full "
                             "width, or fitted to the numbers with --compact).")
    parser.add_argument("--cell-height", type=float, metavar="IN",
                        help="Fix the cell height in inches.")
    parser.add_argument("--row-label-wrap", type=int, metavar="CHARS",
                        help="Wrap row names at this many characters (0 = never; "
                             "default: 14 with --compact, else never).")
    parser.add_argument("--row-label-width", type=float, metavar="IN",
                        help="Width of the row-name gutter in inches "
                             "(default: fitted to the longest row name).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Raster resolution; ignored for PDF.")
    parser.add_argument("--no-baseline-row", action="store_true",
                        help="Omit the per-judge baseline strip above the heatmap.")
    parser.add_argument("--cell-before", action="store_true",
                        help="Print baseline→ablated beneath each delta instead of the "
                             "ablated value alone. Use with --no-baseline-row, where each "
                             "row has its own baseline and no strip states it.")
    parser.add_argument("--no-cell-absolute", action="store_true",
                        help="Show only the delta in each heatmap cell, without "
                             "the post-ablation value beneath it.")
    parser.add_argument("--no-colorbar", action="store_true",
                        help="Drop the shared colour scale under the heatmap.")
    parser.add_argument("--no-footnote", action="store_true",
                        help="Drop the footnote under the figure.")
    parser.add_argument("--row-axis-label",
                        help="Name for what the rows are, e.g. 'Ablation'. "
                             "Also settable as `row_axis_label:` in --config.")
    parser.add_argument("--col-axis-label",
                        help="Name for what the columns are, e.g. 'Judge model'. "
                             "Also settable as `col_axis_label:` in --config.")
    parser.add_argument("--prefix", default="ablation",
                        help="Output filename prefix (default: ablation).")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config) if args.config else None)
    row_labels = cfg.get("row_labels", {}) or {}
    title_overrides = cfg.get("track_titles", {}) or {}
    metric_labels_cfg = cfg.get("metric_labels", {}) or {}
    metrics_cfg = cfg.get("metrics", {}) or {}
    if not isinstance(metrics_cfg, dict):
        raise SystemExit("config `metrics:` must map a setup to a metric key, "
                         "e.g. `pointwise: f1_macro`.")
    unknown = set(metrics_cfg) - set(SETUPS)
    if unknown:
        raise SystemExit(f"config `metrics:` has unknown setup(s): "
                         f"{', '.join(sorted(unknown))}. Valid: {', '.join(SETUPS)}.")
    row_aliases = cfg.get("row_aliases", {}) or {}
    model_labels = cfg.get("model_labels", {}) or {}
    # Row selection/order may live in the config (it is a paper decision, like
    # the labels next to it). --rows still wins for one-off experiments.
    cfg_rows = cfg.get("rows") or None
    if cfg_rows and not isinstance(cfg_rows, list):
        raise SystemExit("config `rows:` must be a list of ablation names.")
    selected_rows = args.rows or cfg_rows
    # Column order is a paper decision like the row order, so it lives here too.
    cfg_models = cfg.get("models") or None
    if cfg_models and not isinstance(cfg_models, list):
        raise SystemExit("config `models:` must be a list of judge model names.")
    # Whether the scale and the footnote appear is a paper decision too, so the
    # config is their home; the flags stay for one-off renders.
    colorbar = bool(cfg.get("colorbar", True)) and not args.no_colorbar
    footnote = cfg.get("footnote")
    if args.no_footnote:
        footnote = ""
    # An empty --row-axis-label/--col-axis-label drops the config's, so a
    # figure that has no room for the title can say so on the command line.
    row_axis_label = (args.row_axis_label if args.row_axis_label is not None
                      else cfg.get("row_axis_label")) or None
    col_axis_label = (args.col_axis_label if args.col_axis_label is not None
                      else cfg.get("col_axis_label")) or None
    # Layout is a paper decision like the labels — a figure that has to fit one
    # column always has to, so the config is its home; the flags stay for
    # one-off renders and only ever turn the tighter layout on.
    stacked = args.stacked or bool(cfg.get("stacked", False))
    compact = args.compact or bool(cfg.get("compact", False))
    model_label_rotation = (args.model_label_rotation
                            if args.model_label_rotation is not None
                            else cfg.get("model_label_rotation"))
    row_label_width = (args.row_label_width if args.row_label_width is not None
                       else cfg.get("row_label_width"))
    cell_width = (args.cell_width if args.cell_width is not None
                  else cfg.get("cell_width"))
    cell_height = (args.cell_height if args.cell_height is not None
                   else cfg.get("cell_height"))
    row_label_wrap = (args.row_label_wrap if args.row_label_wrap is not None
                      else cfg.get("row_label_wrap"))

    tracks = [_parse_track(t) for t in args.track]
    if len(tracks) < 2:
        print("note: only one --track given; the figure will have a single block.",
              file=sys.stderr)

    setups = args.setup or list(SETUPS)
    out_dir = Path(args.out_dir)
    styles = ["heatmap", "forest"] if args.style == "both" else [args.style]

    written_any = False
    for setup in setups:
        loaded = []
        for title, merge_dir in tracks:
            path = _data_path(merge_dir, setup, args.variant)
            if path is None:
                print(f"skipping {setup}: no {args.variant} data for track "
                      f"{title!r} under {merge_dir}", file=sys.stderr)
                loaded = []
                break
            loaded.append((title_overrides.get(title, title),
                           _drop_unsupported_cells(unified_chart_data.load(path))))
        if not loaded:
            continue

        models = args.models or cfg_models
        if not models:
            shared = set(loaded[0][1]["all_models"])
            for _, d in loaded[1:]:
                shared &= set(d["all_models"])
            models = [m for m in loaded[0][1]["all_models"] if m in shared]
        if not models:
            print(f"skipping {setup}: tracks share no judge models.", file=sys.stderr)
            continue

        rows = _common_rows([d for _, d in loaded], selected_rows, row_labels,
                            aliases=row_aliases)
        if not rows:
            print(f"skipping {setup}: no ablation rows to draw.", file=sys.stderr)
            continue

        metric = args.metric or metrics_cfg.get(setup) or DEFAULT_METRIC[setup]
        label = (args.metric_label or metric_labels_cfg.get(metric)
                 or DEFAULT_METRIC_LABEL.get(setup, metric))

        for style in styles:
            fn = (_generate_two_track_heatmap if style == "heatmap"
                  else _generate_two_track_forest)
            # The metric is in the filename, not on the figure: the chart stays
            # slim and the caption says which one, but two metrics for the same
            # setup no longer overwrite each other.
            stem = f"{args.prefix}_{setup}_{args.variant}_{metric}"
            if style == "forest":
                stem += "_forest"
            out_path = out_dir / f"{stem}.{args.formats[0].lstrip('.')}"
            kwargs = dict(tracks=loaded, rows=rows, models=models, metric=metric,
                          metric_label=label, out_path=out_path,
                          model_labels=[model_labels.get(m, m) for m in models],
                          width_in=args.width, formats=args.formats[1:], dpi=args.dpi)
            if style == "heatmap":
                kwargs["stacked"] = stacked
                kwargs["compact"] = compact
                kwargs["model_label_rotation"] = model_label_rotation
                kwargs["row_label_in"] = row_label_width
                kwargs["cell_width_in"] = cell_width
                kwargs["cell_height_in"] = cell_height
                kwargs["row_label_wrap"] = row_label_wrap
                kwargs["baseline_row"] = not args.no_baseline_row
                kwargs["cell_absolute"] = not args.no_cell_absolute
                kwargs["cell_before"] = args.cell_before
                kwargs["colorbar"] = colorbar
                kwargs["row_axis_label"] = row_axis_label
                kwargs["col_axis_label"] = col_axis_label
            if footnote is not None:
                kwargs["footnote"] = footnote
            result = fn(**kwargs)
            if not result:
                reason = ("no CI bounds in the chart data — re-run "
                          "merge_ablation_runs.py --charts-only to regenerate it "
                          "at schema v2" if style == "forest"
                          else f"no finite data for metric {metric!r} — check "
                               "the spelling against the merge data")
                print(f"could not render {setup}/{style}: {reason}", file=sys.stderr)
                continue
            written_any = True
            for p in result:
                print(f"{setup:>9} / {style:<7} → {p}")

    if not written_any:
        raise SystemExit("Nothing was rendered.")


if __name__ == "__main__":
    main()
