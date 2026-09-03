"""
Shared figure logic
===================
What every figure entry point — the notebooks and the remaining scripts — has to
agree on: where a merge's saved chart data lives, which cells are artefacts and
must be blanked, how the same ablation is recognised across differently-named
tracks, and the delta palette.

Layout stays with whoever draws the figure. This module holds only the things
that would silently drift apart if each script kept its own copy, so a figure
regenerated from the notebook and one regenerated from a script cannot disagree
about the numbers they are allowed to show.
"""

from pathlib import Path

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
