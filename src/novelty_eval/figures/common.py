"""
Shared figure logic
===================
What every figure entry point — the notebooks and the remaining scripts — has to
agree on: where a merge's saved chart data lives, which cells are artefacts and
must be blanked, how the same ablation is recognised across differently-named
tracks, and the delta palette.

This module holds the things that would silently drift apart if each entry
point kept its own copy, so a figure regenerated from a notebook and one
regenerated from a script cannot disagree about the numbers they may show.

`TwoTrackFigure` at the bottom is the one full figure that lives here rather
than with its caller: two notebooks draw it over different experiments, so the
drawing is shared and only the wording is theirs.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

# Type 3 fonts are rejected by many venues; 42 embeds TrueType instead.
PDF_RCPARAMS = {"pdf.fonttype": 42, "ps.fonttype": 42}

# Diverging red → off-white → green, for a delta centred on zero.
DELTA_COLORS = ["#d64045", "#f7f4ef", "#3d9970"]

# claude-sonnet-4-5 has no reasoning-effort control, so any number it produces
# under that ablation is an artefact of the harness rather than a result: every
# figure blanks the cell instead of drawing it.
UNSUPPORTED_CELLS = {("low_judge_reasoning", "claude-sonnet-4-5")}

# The same conceptual ablation is named per track *and* per setup when its
# instance set differs: convert_to_plan_form_pairwise_human in one place,
# convert_to_plan_form_pointwise_vanilla in another. Rows are matched on the
# name with these trailing tokens stripped, so one entry covers both setups.
TRACK_SUFFIXES = ("human", "vanilla", "hvh", "vanilla_ai", "ai",
                  "pairwise", "pointwise", "ranking")


def chart_data_path(merge_dir, setup: str, variant: str = "filtered") -> Path:
    """
    Locate unified_<setup>[_<variant>].json under a merge output directory.

    Accepts either the merge directory or its unified_charts/ subdirectory, since
    older merges wrote the files at the top level.
    """
    merge_dir = Path(merge_dir)
    base = merge_dir / "unified_charts" if (merge_dir / "unified_charts").is_dir() \
        else merge_dir
    suffix = "" if variant == "unfiltered" else f"_{variant}"
    return base / f"unified_{setup}{suffix}.json"


def row_key(name: str, suffixes: tuple[str, ...] = TRACK_SUFFIXES) -> str:
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


def drop_unsupported_cells(data: dict) -> dict:
    """Blank cells for judges that cannot honour the ablation being drawn."""
    for name, panel in data.get("panels", {}).items():
        key = row_key(name)
        for ablation, model in UNSUPPORTED_CELLS:
            if key == ablation:
                panel.get("ablation_metrics", {}).pop(model, None)
    return data


def delta_cmap(bad: str = "#e2e8f0"):
    """The diverging delta colormap, with `bad` for cells that have no number."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("delta", DELTA_COLORS, N=256)
    cmap.set_bad(bad)
    return cmap


def save_figure(fig, out_path, formats=None, dpi: int = 300) -> "list[Path]":
    """
    Write a figure to out_path plus any extra formats (by extension swap).

    Returns every path written. PDF output is vector; dpi only affects rasters.
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    targets = [out_path]
    for ext in (formats or []):
        cand = out_path.with_suffix("." + str(ext).lstrip("."))
        if cand not in targets:
            targets.append(cand)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for target in targets:
        fig.savefig(str(target), dpi=dpi, facecolor="white")
        written.append(target)
    plt.close(fig)
    return written


def load_config(path=None) -> dict:
    """
    Read a YAML config into a dict; None means "no config given", i.e. {}.

    Exits with a readable message rather than a traceback — a mistyped path or a
    stray tab in the YAML is a user error, not a bug worth a stack trace.
    """
    if path is None:
        return {}
    path = Path(path)
    try:
        import yaml
    except ImportError:
        raise SystemExit("Reading a config file needs PyYAML installed.")
    if not path.exists():
        raise SystemExit(f"config file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"config is not valid YAML ({path}): {exc}")


# ---------------------------------------------------------------------------
# The two-track ablation figure
# ---------------------------------------------------------------------------
# Rows = ablations, columns = judge models, one column block per data track.
# Drawn twice from one table: once as the change each ablation made, once as
# the score it landed on.

# Shorter judge names for the column headers.
MODEL_LABELS = {"claude-sonnet-4-5": "sonnet-4-5",
                "claude-opus-4-5": "opus-4-5",
                "claude-opus-4-6": "opus-4-6"}

# The baseline strip of a delta figure, a shade of its own: it is not a delta,
# so it must never read as one of zero.
BASELINE_BG = "#f4f6f8"

# Chance on a two-way decision, and the point the red→green scale is white at
# when the figure colours scores rather than changes.
ABSOLUTE_CENTER = 50.0

# White text only on the dark ends of the scale, which is red→green either way.
WHITE_TEXT_BAND = (0.25, 0.82)

# Layout in inches, so a cell is the size it says it is. Every gap on the page
# is one number here: no layout engine to reverse-engineer when it looks wrong.
CELL_IN = 0.44                    # height of one ablation row
STRIP_ROWS = 0.56                 # baseline strip height, in rows — shorter than
                                  # a data row, so it reads as a header
GUTTER_IN = 0.17                  # between the two blocks
MARGIN_IN = {"left": 1.51,        # room for the row names and the axis title
             "right": 0.05, "top": 0.35, "bottom": 0.61}


@dataclass
class TwoTrackFigure:
    """
    One two-track ablation figure: what it says, and how it is drawn.

    Everything that differs between experiments is a field, so nothing in the
    drawing knows which one it is drawing. `table` is the whole data path — one
    tidy row per (track, ablation, judge), everything already in points — and
    the two draw methods are two readings of that table: `draw` colours the
    change an ablation made, `draw_absolute` the score it landed on. Same
    layout either way, so the pair can be read side by side.

    tracks: (block title, merge dir under output/ablation_sweeps, name token),
        left to right. The token fills {track} in a row name.
    rows: (panel name, row label), top to bottom. {setup} and {track} expand,
        for ablations whose instance set differs per track or per setup.
    metrics: the metric to read per setup — the two setups score different
        things. It lands in the filename, never on the figure, so the caption
        has to say which it is.
    """

    root: Path
    figure: str                       # output subdirectory and filename prefix
    tracks: Sequence[tuple[str, str, str]]
    rows: Sequence[tuple[str, str]]
    models: Sequence[str]
    metrics: Mapping[str, str]
    row_axis_label: str
    baseline_label: str = "baseline"  # the un-ablated strip, named like a row
    col_axis_label: str = "Judge model"
    variant: str = "filtered"         # which accuracy report: filtered | unfiltered
    model_labels: Mapping[str, str] = field(
        default_factory=lambda: dict(MODEL_LABELS))
    width: float = 7.1                # figure width, inches

    @property
    def labels(self) -> list:
        """Row labels, top to bottom."""
        return [label for _, label in self.rows]

    @property
    def judges(self) -> list:
        """Column headers, left to right."""
        return [self.model_labels.get(m, m) for m in self.models]

    def panels(self, merge_dir, setup: str) -> dict:
        """The panel dict of one merge's saved chart data."""
        path = chart_data_path(Path(self.root) / "output/ablation_sweeps" / merge_dir,
                               setup, self.variant)
        return json.loads(path.read_text())["panels"]

    def table(self, setup: str):
        """One row per (track, ablation, judge). Scores and deltas are in points."""
        import pandas as pd

        metric = self.metrics[setup]
        recs = []
        for title, merge_dir, token in self.tracks:
            ps = self.panels(merge_dir, setup)
            for name, label in self.rows:
                panel = ps.get(name.format(setup=setup, track=token))
                for model in self.models:
                    if panel is None or (name, model) in UNSUPPORTED_CELLS:
                        base = abl = sig = None
                    else:
                        base = panel["baseline_metrics"].get(model, {}).get(metric)
                        abl = panel["ablation_metrics"].get(model, {}).get(metric)
                        # Bootstrap results are keyed "<model>||<metric>" on disk.
                        sig = panel["bootstrap_results"].get(
                            f"{model}||{metric}", {}).get("significant")
                    recs.append({
                        "track": title,
                        "ablation": label,
                        "judge": self.model_labels.get(model, model),
                        "baseline": None if base is None else base * 100,
                        "ablated": None if abl is None else abl * 100,
                        "delta": None if base is None or abl is None
                                 else (abl - base) * 100,
                        "significant": bool(sig),
                    })
        return pd.DataFrame(recs)

    def draw(self, setup: str) -> Path:
        """Draw the delta reading of one setup: save PDF + PNG, return the PNG."""
        return self._draw(setup, absolute=False)

    def draw_absolute(self, setup: str, pad: float = 2.0,
                      center: float = ABSOLUTE_CENTER) -> Path:
        """
        Draw the absolute reading of one setup: the same grid on the same
        red→green scale, but a cell is the score itself rather than the change,
        and the baseline is an ordinary row of it.

        `center` is the score the scale is white at, red below and green above;
        `pad` is the headroom in points left past the extreme cells, so neither
        end of the palette is reached flat.
        """
        return self._draw(setup, absolute=True, pad=pad, center=center)

    def _draw(self, setup: str, absolute: bool, pad: float = 2.0,
              center: float = ABSOLUTE_CENTER) -> Path:
        """
        One `pcolormesh` per track, then the cell text on top of it.

        Every cell prints its number, so colour is never the only encoding, and
        both readings share one scale across the blocks: a colour means the same
        thing left and right.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        plt.rcParams.update({"font.family": "sans-serif", **PDF_RCPARAMS})
        df = self.table(setup)
        labels, judges = self.labels, self.judges
        cmap = delta_cmap()

        if absolute:
            # One scale for every number on the page, baseline row included, so
            # a shade means the same score in either block. Kept symmetric about
            # `center`, which is what makes red mean "below chance" rather than
            # "low in whatever range this figure happens to cover".
            vals = pd.concat([df["ablated"], df["baseline"]]).astype(float)
            span = np.nanmax(np.abs(vals - center)) + pad
            vmin, vmax = center - span, center + span
        else:
            span = max(1.0, np.nanmax(np.abs(df["delta"].astype(float))))
            vmin, vmax = -span, span

        def text_colour(v):
            """Cell text: white only on the dark ends of the scale."""
            shade = (v - vmin) / (vmax - vmin)
            dark = shade < WHITE_TEXT_BAND[0] or shade > WHITE_TEXT_BAND[1]
            return "white" if dark else "#1a1a1a"

        # The delta reading measures against the baseline, so the baseline sits
        # outside the grid as a strip; the absolute reading has one kind of
        # number throughout, so there the baseline is simply the top row.
        strip_rows = 1.0 if absolute else STRIP_ROWS
        strip_top = 1 - strip_rows

        grid_in = CELL_IN * (len(labels) + strip_rows)
        fig_h = grid_in + MARGIN_IN["top"] + MARGIN_IN["bottom"]
        blocks_in = self.width - MARGIN_IN["left"] - MARGIN_IN["right"]
        block_in = (blocks_in - GUTTER_IN * (len(self.tracks) - 1)) / len(self.tracks)

        fig = plt.figure(figsize=(self.width, fig_h))
        gs = fig.add_gridspec(
            1, len(self.tracks), wspace=GUTTER_IN / block_in,
            left=MARGIN_IN["left"] / self.width,
            right=1 - MARGIN_IN["right"] / self.width,
            top=1 - MARGIN_IN["top"] / fig_h, bottom=MARGIN_IN["bottom"] / fig_h)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(self.tracks))]

        for ax, (title, _, _) in zip(axes, self.tracks):
            block = df[df["track"] == title].set_index(["ablation", "judge"])
            grid = lambda col: block[col].unstack().reindex(index=labels,
                                                            columns=judges)
            delta, ablated, sig = grid("delta"), grid("ablated"), grid("significant")
            # Every ablation in a track shares one baseline run, so it is one row
            # of its own rather than a number repeated down every row.
            base = grid("baseline").bfill().iloc[0]

            mesh = dict(cmap=cmap, vmin=vmin, vmax=vmax,
                        edgecolors="white", linewidth=1.2)
            if absolute:
                ax.pcolormesh(np.arange(len(judges) + 1),
                              np.arange(len(labels) + 2),
                              np.ma.masked_invalid(np.vstack(
                                  [base.to_numpy(float), ablated.to_numpy(float)])),
                              **mesh)
            else:
                ax.pcolormesh(np.arange(len(judges) + 1),
                              np.arange(1, len(labels) + 2),
                              np.ma.masked_invalid(delta.to_numpy(float)), **mesh)
                # Outside the grid, in a shade of its own, so the baseline never
                # reads as a delta of zero.
                ax.axhspan(strip_top, 0.995, facecolor=BASELINE_BG, zorder=0)
                ax.vlines(range(1, len(judges)), strip_top, 0.995,
                          color="white", linewidth=1.2)
            ax.set_xlim(0, len(judges))
            ax.set_ylim(len(labels) + 1, strip_top)      # baseline on top

            for j, judge in enumerate(judges):
                ax.text(j + 0.5, (strip_top + 1) / 2, f"{base[judge]:.1f}",
                        ha="center", va="center",
                        fontsize=7.0 if absolute else 6.4, fontweight="bold",
                        color=text_colour(base[judge]) if absolute else "#333333")
                for i, label in enumerate(labels):
                    d, a = delta.loc[label, judge], ablated.loc[label, judge]
                    lead = a if absolute else d      # the number that colours the cell
                    if not np.isfinite(float(lead if lead is not None else np.nan)):
                        ax.text(j + 0.5, i + 1.5, "—", ha="center", va="center",
                                fontsize=6, color="#999999")
                        continue
                    colour = text_colour(lead)
                    if absolute:
                        # The score alone. Significance is a claim about the
                        # change, so it is marked where the change is drawn.
                        ax.text(j + 0.5, i + 1.5, f"{a:.1f}", ha="center",
                                va="center", fontsize=7.0, fontweight="bold",
                                color=colour)
                        continue
                    star = "*" if sig.loc[label, judge] else ""
                    ax.text(j + 0.5, i + 1.32, f"{d:+.1f}{star}", ha="center",
                            va="center", fontsize=6.6, fontweight="bold",
                            color=colour)
                    # Italic, so the absolute never reads as a second delta.
                    ax.text(j + 0.5, i + 1.68, f"({a:.1f})", ha="center",
                            va="center", fontsize=5.4, style="italic",
                            color=colour, alpha=0.72)

            ax.set_xticks(np.arange(len(judges)) + 0.5, judges, fontsize=6.2,
                          rotation=38, ha="right", rotation_mode="anchor")
            ax.set_yticks(
                [(strip_top + 1) / 2] + [i + 1.5 for i in range(len(labels))],
                [self.baseline_label] + labels, fontsize=6.6)
            ax.set_title(title, fontsize=8, fontweight="bold", pad=8.5,
                         bbox=dict(boxstyle="round,pad=0.35", fc="#eef1f5",
                                   ec="none"))
            ax.tick_params(length=0, pad=2)
            for spine in ax.spines.values():
                spine.set_visible(False)

        # The row names, and both axis titles, belong to the left-hand block only.
        for ax in axes[1:]:
            ax.tick_params(labelleft=False)
        if not absolute:
            # The strip is labelled like a row, but lighter: it is not an
            # ablation. In the absolute figure it *is* an ordinary row.
            axes[0].get_yticklabels()[0].set(style="italic", color="#555555")
        axes[0].set_ylabel(self.row_axis_label, fontsize=7.2, fontweight="bold",
                           color="#333333", labelpad=5)
        fig.text((MARGIN_IN["left"] + blocks_in / 2) / self.width, 0.12 / fig_h,
                 self.col_axis_label, ha="center", va="bottom", fontsize=7.2,
                 fontweight="bold", color="#333333")

        stem = f"{self.figure}_{setup}_{self.variant}_{self.metrics[setup]}"
        out_dir = Path(self.root) / "output/figures" / self.figure
        out_path = out_dir / f"{stem}{'_absolute' if absolute else ''}.pdf"
        save_figure(fig, out_path, formats=["png"])
        return out_path.with_suffix(".png")
