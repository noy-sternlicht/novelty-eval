"""
Heatmap visualization for ablation delta comparisons.

Generates matplotlib figures showing metric deltas (ablation − baseline)
across models, with optional bootstrap significance annotations.
"""
import numpy as np
from pathlib import Path

# Type 3 fonts are rejected by many venues; 42 embeds TrueType instead.
_PDF_RCPARAMS = {"pdf.fonttype": 42, "ps.fonttype": 42}


def _generate_overview_heatmap(
    ablation_labels: list[str],
    metric_specs: list[tuple[str, str]],
    delta_matrix,
    sig_matrix,
    out_path: Path,
    model: str = "",
    n_pointwise: int = 0,
    base_matrix=None,
    abs_matrix=None,
    reliable_matrix=None,
) -> "Path | None":
    """
    Overview heatmap: rows = ablations, columns = metrics.

    Rendered as two side-by-side GridSpec panels — left for pointwise metrics,
    right for pairwise — sharing the y-axis (ablation labels).  Cells show
    delta vs baseline with (base → ablation) below; * marks a significant delta,
    ? marks a cell where the bootstrap CI was unreliable.

    Args:
        ablation_labels: display name for each ablation row.
        metric_specs: [(display_label, setup), ...] ordered same as delta_matrix columns.
        delta_matrix: (n_ablations × n_metrics) float array, NaN = missing.
        sig_matrix: (n_ablations × n_metrics) bool array, True = significant.
        out_path: where to write the PNG.
        model: judge-model name, used in the figure title.
        n_pointwise: number of leading columns that belong to pointwise setup.
        base_matrix: (n_ablations × n_metrics) baseline absolute values (optional).
        abs_matrix: (n_ablations × n_metrics) ablation absolute values (optional).
        reliable_matrix: (n_ablations × n_metrics) bool array, False = CI unreliable (optional).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.gridspec import GridSpec
    except ImportError:
        return None

    n_abl = len(ablation_labels)
    n_metrics = len(metric_specs)
    n_pairwise = n_metrics - n_pointwise
    if n_abl == 0 or n_metrics == 0:
        return None

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg_delta", ["#d64045", "#f7f4ef", "#3d9970"], N=256
    )
    cmap.set_bad(color="#e2e8f0")

    abs_max = float(np.nanmax(np.abs(delta_matrix))) if not np.all(np.isnan(delta_matrix)) else 0.1
    abs_max = max(abs_max, 0.01)
    vmin, vmax = -abs_max, abs_max

    fig_w = max(6.0, n_metrics * 1.9 + 2.5)
    fig_h = max(3.0, n_abl * 0.7 + 2.2)

    has_both = n_pointwise > 0 and n_pairwise > 0
    if has_both:
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = GridSpec(1, 2, width_ratios=[n_pointwise, n_pairwise], figure=fig, wspace=0.06)
        ax_ptw = fig.add_subplot(gs[0])
        ax_pair = fig.add_subplot(gs[1], sharey=ax_ptw)
        sections = [
            (ax_ptw, 0, n_pointwise, "Pointwise"),
            (ax_pair, n_pointwise, n_metrics, "Pairwise"),
        ]
    else:
        fig, ax_only = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
        sections = [(ax_only, 0, n_metrics, "")]

    show_transition = base_matrix is not None and abs_matrix is not None

    def _render_section(ax, col_start, col_end, section_label):
        ax.set_facecolor("white")
        n_cols = col_end - col_start
        sec_delta    = delta_matrix[:, col_start:col_end]
        sec_sig      = sig_matrix[:, col_start:col_end]
        sec_reliable = reliable_matrix[:, col_start:col_end] if reliable_matrix is not None else None
        sec_base = base_matrix[:, col_start:col_end] if show_transition else None
        sec_abs  = abs_matrix[:, col_start:col_end]  if show_transition else None
        col_labels = [metric_specs[j][0] for j in range(col_start, col_end)]

        ax.imshow(sec_delta, cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect="auto", interpolation="nearest")

        for i in range(n_abl):
            for j in range(n_cols):
                v = sec_delta[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#bbbbbb")
                else:
                    norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    text_color = "white" if norm_v < 0.25 or norm_v > 0.82 else "#1a1a1a"
                    if sec_reliable is not None and not sec_reliable[i, j]:
                        sig_marker = "?"
                    elif sec_sig[i, j]:
                        sig_marker = "*"
                    else:
                        sig_marker = ""
                    cell_text = f"{'+' if v >= 0 else ''}{v * 100:.1f}%{sig_marker}"
                    y_delta = i - 0.13 if show_transition else i
                    ax.text(j, y_delta, cell_text, ha="center", va="center",
                            fontsize=9, color=text_color, fontweight="bold")
                    if show_transition:
                        bv = sec_base[i, j]
                        av = sec_abs[i, j]
                        if not (np.isnan(bv) or np.isnan(av)):
                            trans = f"({bv * 100:.1f} → {av * 100:.1f})"
                            ax.text(j, i + 0.18, trans, ha="center", va="center",
                                    fontsize=7, color=text_color, alpha=0.88,
                                    fontstyle="italic")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=9, rotation=0, ha="center")
        ax.tick_params(length=0)
        if section_label:
            ax.set_title(section_label, fontsize=11, fontweight="bold", pad=18)

        for x in range(n_cols + 1):
            ax.axvline(x - 0.5, color="white", linewidth=1.5)
        for y in range(n_abl + 1):
            ax.axhline(y - 0.5, color="white", linewidth=1.5)

        if col_start == 0:
            ax.set_yticks(range(n_abl))
            ax.set_yticklabels(ablation_labels, fontsize=9)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    for ax, col_start, col_end, section_label in sections:
        _render_section(ax, col_start, col_end, section_label)

    fig.suptitle(model or "Ablation Overview", fontsize=12, fontweight="bold", y=1.06)
    fig.text(
        0.5, -0.01,
        "* significant (95% BCa CI); ? = CI unreliable — corpus-level paired bootstrap, n=100,000",
        ha="center", va="top", fontsize=7.5, color="#666666", style="italic",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _generate_unified_heatmap(
    setup: str,
    panels: list[dict],
    metric_labels: dict[str, str],
    all_models: list[str],
    out_path: "Path",
    formats=None,
) -> "Path | None":
    """
    Unified side-by-side heatmap: one panel per ablation, all on one figure.

    Each panel shows rows = judge models, columns = metrics (delta vs baseline),
    using a **shared** colour scale so cells are visually comparable across panels.

    panels: list of dicts, each with:
        desc (str)               — ablation description used as panel title
        baseline_metrics (dict)  — model → {metric_key → value}
        ablation_metrics (dict)  — model → {metric_key → value}
        bootstrap_results (dict) — (model, metric_key) → bootstrap result dict
    all_models: ordered list of judge models (rows, same for every panel).
    metric_labels: ordered dict of metric_key → display label (columns per panel).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return None

    if not panels or not metric_labels or not all_models:
        return None

    n_panels  = len(panels)
    n_models  = len(all_models)
    metric_keys = list(metric_labels.keys())
    n_metrics = len(metric_keys)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        **_PDF_RCPARAMS,
    })

    # ── Build one matrix per panel ────────────────────────────────────────────
    all_delta_vals: list[float] = []
    panel_data: list[tuple] = []  # (delta, base, abs_, sig, relbl, bs)
    for panel in panels:
        bm_all = panel.get("baseline_metrics", {})
        am_all = panel.get("ablation_metrics", {})
        bs     = panel.get("bootstrap_results", {})

        delta = np.full((n_models, n_metrics), np.nan)
        base  = np.full((n_models, n_metrics), np.nan)
        abs_  = np.full((n_models, n_metrics), np.nan)
        sig   = np.zeros((n_models, n_metrics), dtype=bool)
        relbl = np.ones((n_models, n_metrics),  dtype=bool)

        for i, model in enumerate(all_models):
            bm = bm_all.get(model, {})
            am = am_all.get(model, {})
            for j, mk in enumerate(metric_keys):
                bv = bm.get(mk)
                av = am.get(mk)
                if bv is not None and av is not None:
                    d = av - bv
                    delta[i, j] = d
                    base[i, j]  = bv
                    abs_[i, j]  = av
                    all_delta_vals.append(d)
                    bs_e = bs.get((model, mk), {})
                    sig[i, j]   = bool(bs_e.get("significant", False))
                    relbl[i, j] = bool(bs_e.get("reliable", True))

        panel_data.append((delta, base, abs_, sig, relbl, bs))

    # ── Shared colour scale across all panels ─────────────────────────────────
    abs_max = max(abs(v) for v in all_delta_vals) if all_delta_vals else 0.1
    abs_max = max(abs_max, 0.01)
    vmin, vmax = -abs_max, abs_max

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg_delta", ["#d64045", "#f7f4ef", "#3d9970"], N=256
    )
    cmap.set_bad(color="#e2e8f0")

    # ── Figure layout ─────────────────────────────────────────────────────────
    panel_w = max(1.4, n_metrics * 1.6 + 0.2)

    # Wrap titles to fit panel width before creating the figure so we can
    # reserve the correct amount of vertical space for multi-line titles.
    import textwrap as _tw
    _chars_per_line = max(12, int(panel_w * 7.0))  # ~7 chars per inch at fontsize 9
    wrapped_titles = [_tw.fill(p["desc"], width=_chars_per_line) for p in panels]
    _max_title_lines = max((t.count("\n") + 1 for t in wrapped_titles), default=1)

    fig_w   = panel_w * n_panels + 2.2 + (n_panels - 1) * 0.15
    # Reserve 0.30 in per extra title line beyond the first
    fig_h   = max(2.5, n_models * 0.75 + 2.2 + max(0, _max_title_lines - 1) * 0.30)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(fig_w, fig_h),
        sharey=True,
        facecolor="white",
        squeeze=False,
    )
    axes = axes[0]  # shape → (n_panels,)

    has_bootstrap = any(bool(p.get("bootstrap_results")) for p in panels)

    for p_idx, (ax, panel) in enumerate(zip(axes, panels)):
        ax.set_facecolor("white")
        delta, base, abs_, sig, relbl, bs = panel_data[p_idx]

        ax.imshow(delta, cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect="auto", interpolation="nearest")

        for i in range(n_models):
            for j in range(n_metrics):
                v   = delta[i, j]
                mk  = metric_keys[j]
                mdl = all_models[i]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=8, color="#bbbbbb")
                else:
                    norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    tc = "white" if norm_v < 0.25 or norm_v > 0.82 else "#1a1a1a"
                    bs_e = bs.get((mdl, mk), {})
                    star = ("?" if not relbl[i, j]
                            else ("*" if sig[i, j] else ""))
                    bv, av = base[i, j], abs_[i, j]
                    if "n_ties" in mk or "support" in mk:
                        top = f"{'+' if v > 0 else ''}{v:.1f}{star}"
                        bot = f"({bv:.1f}→{av:.1f})"
                    else:
                        top = f"{'+' if v >= 0 else ''}{v * 100:.1f}%{star}"
                        bot = f"({bv * 100:.1f}→{av * 100:.1f})"
                    ax.text(j, i - 0.13, top, ha="center", va="center",
                            fontsize=8, color=tc, fontweight="bold")
                    if not (np.isnan(bv) or np.isnan(av)):
                        ax.text(j, i + 0.18, bot, ha="center", va="center",
                                fontsize=6.5, color=tc, alpha=0.88,
                                fontstyle="italic")

        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels(
            [metric_labels[k] for k in metric_keys],
            fontsize=8, rotation=0, ha="center",
        )
        ax.tick_params(length=0)
        ax.set_title(wrapped_titles[p_idx], fontsize=9, fontweight="bold", pad=6)

        for x in range(n_metrics + 1):
            ax.axvline(x - 0.5, color="white", linewidth=1.5)
        for y in range(n_models + 1):
            ax.axhline(y - 0.5, color="white", linewidth=1.5)

        if p_idx == 0:
            ax.set_yticks(range(n_models))
            ax.set_yticklabels(all_models, fontsize=9)
            ax.set_ylabel("Judge Model", fontsize=11, fontweight="bold", labelpad=8)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Push suptitle above the tallest (possibly multi-line) panel title
    _suptitle_y = 1.03 + max(0, _max_title_lines - 1) * 0.04
    fig.suptitle(
        f"{setup.capitalize()} — Unified Ablation Comparison",
        fontsize=12, fontweight="bold", y=_suptitle_y,
    )
    if has_bootstrap:
        fig.text(
            0.5, -0.01,
            "* significant (95% BCa CI); ? = CI unreliable — "
            "corpus-level paired bootstrap, n=100,000",
            ha="center", va="top", fontsize=7.5, color="#666666", style="italic",
        )

    plt.subplots_adjust(wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets = [out_path]
    for ext in (formats or []):
        cand = out_path.with_suffix("." + str(ext).lstrip("."))
        if cand not in targets:
            targets.append(cand)
    for target in targets:
        fig.savefig(str(target), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _generate_aggregated_delta_heatmap(
    setup: str,
    metric_labels: dict[str, str],
    models: list[str],
    baseline_metrics: dict[str, dict],
    ablation_metrics: dict[str, dict],
    out_path: Path,
    after_desc: str = "",
    before_desc: str = "",
    ablation_name: str = "",
    bootstrap_results: dict[tuple[str, str], dict] = None,
) -> Path | None:
    """
    Generate a single heatmap for a setup, showing deltas for all metrics across all models.
    Rows = Models, Columns = Metrics.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return None

    if not models or not metric_labels:
        return None

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    n_models = len(models)
    metric_keys = list(metric_labels.keys())
    n_metrics = len(metric_keys)

    # Build delta matrix (models x metrics), absolute value matrix, and baseline matrix
    delta_mat = np.zeros((n_models, n_metrics))
    abs_mat = np.zeros((n_models, n_metrics))
    base_mat = np.zeros((n_models, n_metrics))
    for i, model in enumerate(models):
        bm = baseline_metrics.get(model, {})
        am = ablation_metrics.get(model, {})
        for j, mk in enumerate(metric_keys):
            bv = bm.get(mk)
            av = am.get(mk)
            if bv is not None and av is not None:
                delta_mat[i, j] = av - bv
                abs_mat[i, j] = av
                base_mat[i, j] = bv
            else:
                delta_mat[i, j] = np.nan
                abs_mat[i, j] = np.nan
                base_mat[i, j] = np.nan

    display_mat = delta_mat
    display_abs = abs_mat
    display_base = base_mat
    display_keys = metric_keys
    display_labels = metric_labels
    n_display = len(display_keys)

    cmap_delta = mcolors.LinearSegmentedColormap.from_list(
        "rg_delta", ["#d64045", "#f7f4ef", "#3d9970"], N=256
    )
    cmap_delta.set_bad(color="#e2e8f0")

    # Symmetric scale around 0, based on max absolute delta across all metrics
    delta_abs_max = float(np.nanmax(np.abs(delta_mat))) if not np.all(np.isnan(delta_mat)) else 0.1
    delta_abs_max = max(delta_abs_max, 0.01)
    vmin, vmax = -delta_abs_max, delta_abs_max

    # Dynamically size figure based on number of models and metrics
    fig_w = max(4.5, n_display * 1.6 + 1.5)
    fig_h = max(2.5, n_models * 0.75 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")

    plt.subplots_adjust(top=0.92, bottom=0.10)

    im = ax.imshow(display_mat, cmap=cmap_delta, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")

    # Add text labels to each cell
    for i in range(n_models):
        for j in range(n_display):
            v = display_mat[i, j]
            abs_v = display_abs[i, j]
            base_v = display_base[i, j]
            model = models[i]
            mk = display_keys[j]

            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="#bbbbbb")
            else:
                norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                # Use white text on dark backgrounds
                text_color = "white" if norm_v < 0.25 or norm_v > 0.82 else "#1a1a1a"

                sig_stars = ""
                if bootstrap_results and (model, mk) in bootstrap_results:
                    res = bootstrap_results[(model, mk)]
                    if not res.get("reliable", True):
                        sig_stars = "?"
                    elif res.get("significant"):
                        sig_stars = "*"

                if "n_ties" in mk or "support" in mk:
                    delta_label = f"{'+' if v > 0 else ''}{v:.1f}{sig_stars}"
                    abs_label = f"({base_v:.1f} → {abs_v:.1f})"
                else:
                    delta_label = f"{'+' if v > 0 else ''}{v*100:.1f}%{sig_stars}"
                    abs_label = f"({base_v*100:.1f} → {abs_v*100:.1f})"

                # Delta (large)
                ax.text(j, i - 0.12, delta_label, ha="center", va="center",
                        fontsize=10, color=text_color, fontweight="bold")
                # Result (smaller, italic transition)
                ax.text(j, i + 0.18, abs_label, ha="center", va="center",
                        fontsize=7.0, color=text_color, alpha=0.9, fontstyle="italic")

    # Setup axes
    ax.set_xticks(range(n_display))
    ax.set_xticklabels([display_labels[k] for k in display_keys],
                       fontsize=10, rotation=0, ha="center")
    ax.tick_params(length=0)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models, fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("Judge Model", fontsize=13, fontweight="bold", labelpad=8)

    # Grid lines to separate cells
    for x in range(n_display + 1):
        ax.axvline(x - 0.5, color="white", linewidth=2.0)
    for y in range(n_models + 1):
        ax.axhline(y - 0.5, color="white", linewidth=2.0)

    if bootstrap_results:
        fig.text(
            0.5, 0.0,
            "* significant (95% BCa CI); ? = CI unreliable — corpus-level paired bootstrap, n=100,000",
            ha="center", va="top",
            fontsize=8, color="#666666", style="italic",
            transform=fig.transFigure,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Two-track figures (one column block per data type, e.g. human-only vs
# human+generated). Rows are ablations rather than judge models, so the figure
# grows downward as ablations are added instead of sideways — which is what
# makes it fit a paper column.
# ---------------------------------------------------------------------------

_DELTA_CMAP_COLORS = ["#d64045", "#f7f4ef", "#3d9970"]


def _save_figure(fig, out_path: "Path", formats=None, dpi: int = 300) -> "list[Path]":
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


def _collect_track_matrices(track_data: dict, rows: list, metric: str,
                            models: list[str], track_index: int):
    """
    Build the per-cell arrays for one track's column block.

    rows: list of (display_label, canonical_per_track_tuple). The canonical name
    may differ between tracks (e.g. ablations suffixed by track), so each row
    carries one name per track and `track_index` selects it.

    Returns a dict of (n_rows x n_models) arrays: delta, base, ci_low, ci_high
    (NaN where absent) plus boolean sig / rel.
    """
    n_r, n_c = len(rows), len(models)
    out = {k: np.full((n_r, n_c), np.nan)
           for k in ("delta", "base", "abl", "ci_low", "ci_high")}
    out["sig"] = np.zeros((n_r, n_c), dtype=bool)
    out["rel"] = np.ones((n_r, n_c), dtype=bool)

    panels = track_data.get("panels", {})
    for i, (_, canonicals) in enumerate(rows):
        name = canonicals[track_index]
        panel = panels.get(name) if name else None
        if not panel:
            continue
        for j, model in enumerate(models):
            bv = (panel.get("baseline_metrics", {}).get(model) or {}).get(metric)
            av = (panel.get("ablation_metrics", {}).get(model) or {}).get(metric)
            if bv is None or av is None:
                continue
            out["delta"][i, j] = (av - bv) * 100.0
            out["base"][i, j] = bv * 100.0
            out["abl"][i, j] = av * 100.0
            bs = (panel.get("bootstrap_results", {}) or {}).get((model, metric), {})
            out["sig"][i, j] = bool(bs.get("significant", False))
            out["rel"][i, j] = bool(bs.get("reliable", True))
            lo, hi = bs.get("ci_low"), bs.get("ci_high")
            if lo is not None:
                out["ci_low"][i, j] = lo * 100.0
            if hi is not None:
                out["ci_high"][i, j] = hi * 100.0
    return out


def _text_width_in(labels, fontsize: float, **text_kw) -> float:
    """Width of the widest label, in inches, as the renderer will draw it."""
    import matplotlib.pyplot as plt

    labels = [str(x) for x in labels if str(x)]
    if not labels:
        return 0.0
    fig = plt.figure(figsize=(1, 1))
    try:
        rend = fig.canvas.get_renderer()
        widest = 0.0
        for label in labels:
            t = fig.text(0, 0, label, fontsize=fontsize, **text_kw)
            widest = max(widest, t.get_window_extent(rend).width / fig.dpi)
            t.remove()
        return widest
    finally:
        plt.close(fig)


def _generate_two_track_heatmap(
    tracks: list,
    rows: list,
    models: list[str],
    metric: str,
    metric_label: str,
    out_path: "Path",
    model_labels: list[str] | None = None,
    baseline_row: bool = True,
    cell_absolute: bool = True,
    cell_before: bool = False,
    colorbar: bool = True,
    row_axis_label: str | None = None,
    col_axis_label: str | None = None,
    width_in: float = 7.1,
    stacked: bool = False,
    compact: bool = False,
    cell_width_in: float | None = None,
    cell_height_in: float | None = None,
    row_label_wrap: int | None = None,
    model_label_rotation: float | None = None,
    row_label_in: float | None = None,
    formats=None,
    dpi: int = 300,
    footnote: str | None = None,
) -> "list[Path] | None":
    """
    Transposed two-track heatmap: rows = ablations, columns = judge models,
    one block per track, shared colour scale and one shared colourbar.

    Args:
        tracks: [(display_title, loaded_chart_data), ...] — one block each.
        rows: [(display_label, (canonical_for_track0, canonical_for_track1, ...)), ...]
            in top-to-bottom order. A canonical name of None (or one absent from
            that track's data) renders as an empty cell.
        models: judge models, left-to-right within every block.
        metric: metric key to draw (a single column per judge).
        metric_label: colourbar label.
        baseline_row: show each judge's baseline once above the grid. The
            baseline is constant per judge within a track, so repeating it in
            every cell would be redundant.
        cell_absolute: print the post-ablation value in small type beneath each
            delta, so a cell can be read without adding it to the baseline strip.
        colorbar: draw the shared colour scale under the blocks. Every cell is
            labelled, so colour is redundant encoding — drop it where the caption
            carries the scale instead.
        row_axis_label: name for what the rows are, e.g. "Ablation" — rotated in
            the left gutter. None omits it and its width.
        col_axis_label: name for what the columns are, e.g. "Judge model" —
            centred under the blocks. None omits it and its height.
        stacked: put the track blocks one above the other instead of side by
            side, so the figure is narrow and tall — a one-column figure. Every
            block keeps its own judge names, so each reads on its own without
            counting columns down to the bottom of the stack.
        compact: tighter cells, gutters and type throughout, and cells only as
            wide as the numbers in them — so `width_in` becomes a maximum the
            figure need not spend. Pair it with a one-column width, where the
            default spacing wastes the page.
        cell_width_in / cell_height_in: fix a cell's size in inches instead of
            fitting it. A width wider than the page allows is clipped back to
            what fits.
        model_label_rotation: angle for the judge names under the grid. None
            picks 38 degrees normally and 0 in compact mode, where the names are
            short enough to sit flat and the slanted band costs more height than
            it saves width.
        row_label_in: width of the left gutter holding the row names. None
            measures the longest one and fits the gutter to it.
        row_label_wrap: wrap row names onto further lines at this many
            characters, so a long name does not buy a gutter wider than the grid
            it labels. None wraps at 14 in compact mode and not at all
            otherwise; 0 never wraps.
        footnote: text under the figure. None uses the significance/baseline
            default; "" omits the footnote and its height.
        formats: extra output formats, e.g. ["png"] alongside a .pdf out_path.

    Returns the paths written, or None if matplotlib is unavailable or there is
    no finite data to draw.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return None

    if not tracks or not rows or not models:
        return None

    _model_names = list(model_labels) if model_labels else list(models)
    if row_label_wrap is None:
        row_label_wrap = 14 if compact else 0
    if row_label_wrap:
        import textwrap
        _row_names = ["\n".join(textwrap.wrap(str(r[0]), row_label_wrap)) or str(r[0])
                      for r in rows]
    else:
        _row_names = [str(r[0]) for r in rows]
    blocks = [(title, _collect_track_matrices(data, rows, metric, models, ti))
              for ti, (title, data) in enumerate(tracks)]
    finite = np.concatenate([b["delta"][np.isfinite(b["delta"])] for _, b in blocks])
    if finite.size == 0:
        return None

    vmax = max(float(np.abs(finite).max()), 1.0)
    vmin = -vmax
    cmap = mcolors.LinearSegmentedColormap.from_list("rg_delta", _DELTA_CMAP_COLORS, N=256)
    cmap.set_bad(color="#e2e8f0")

    n_r, n_c, n_b = len(rows), len(models), len(blocks)
    if footnote is None:
        footnote = ("* significant at 95% (BCa CI, paired bootstrap, n=100,000); "
                    "? = CI unreliable.\nEach cell is the change vs. that judge's "
                    "baseline")
        if cell_absolute:
            footnote += (", with baseline→ablated beneath it."
                         if cell_before else ", with the ablated value beneath it.")
        else:
            footnote += "."

    plt.rcParams.update({"font.family": "sans-serif", **_PDF_RCPARAMS})

    # Two spacing regimes. `compact` shaves roughly a third off the height by
    # tightening every gutter and dropping each font about half a point — the
    # figure still has to be legible at one-column width, so nothing here goes
    # below ~5pt.
    if compact:
        fs_delta, fs_abs, fs_row, fs_col, fs_title, fs_base = 6.0, 5.0, 6.2, 5.8, 7.0, 5.8
        # Two lines of type (6pt + 5pt), their leading, and room above and
        # below so neither line sits on the cell edge.
        cell_h = 0.30 if cell_absolute else 0.20
        gap_in, top_gap_in, title_in, base_h_in = 0.08, 0.06, 0.16, 0.15
        grid_lw, title_pad, tick_pad = 0.8, 0.18, 1.2
        # Fractions of a cell: the two lines sit closer in the shorter cell.
        cell_line_off = 0.21
    else:
        fs_delta, fs_abs, fs_row, fs_col, fs_title, fs_base = 6.4, 5.4, 6.8, 6.0, 7.8, 6.0
        # A second line of type per cell needs the extra height, or the two
        # numbers crowd the cell edges.
        cell_h = 0.44 if cell_absolute else 0.38
        gap_in, top_gap_in, title_in, base_h_in = 0.16, 0.07, 0.25, 0.24
        cell_line_off = 0.18
        grid_lw, title_pad, tick_pad = 1.0, 0.30, 1.5
    rot = float(model_label_rotation) if model_label_rotation is not None \
        else (0.0 if compact else 38.0)
    # The title sits in a rounded chip whose padding is set in fractions of its
    # font size. Counting that padding here keeps `top_gap_in` the gap you see
    # between the chip and the grid, rather than the one to the text inside it.
    chip_pad_in = title_pad * fs_title / 72.0

    def _cell_text():
        """Every string a cell has to hold, so the grid can be fitted to them."""
        deltas, absolutes = [], []
        for _, m in blocks:
            for i in range(n_r):
                for j in range(n_c):
                    v, av, bv = m["delta"][i, j], m["abl"][i, j], m["base"][i, j]
                    if not np.isfinite(v):
                        continue
                    star = "?" if not m["rel"][i, j] else ("*" if m["sig"][i, j] else "")
                    deltas.append(f"{v:+.1f}{star}")
                    if cell_absolute and np.isfinite(av):
                        absolutes.append(f"({bv:.1f}\u2192{av:.1f})"
                                         if cell_before and np.isfinite(bv)
                                         else f"({av:.1f})")
        return deltas, absolutes

    def _fitted_cell_width_in():
        """Widest thing in a column of cells, plus a little breathing room."""
        deltas, absolutes = _cell_text()
        widest = max(
            _text_width_in(deltas, fs_delta, fontweight="bold"),
            _text_width_in(absolutes, fs_abs, style="italic"),
            _text_width_in([f"{b:.1f}" for _, m in blocks
                            for b in m["base"].ravel() if np.isfinite(b)],
                           fs_base, fontweight="bold") if baseline_row else 0.0,
            # A flat judge name sits under its own column and has to fit it too;
            # a slanted one runs past the column by design.
            _text_width_in(_model_names, fs_col) if abs(rot) < 1.0 else 0.0,
        )
        # Enough clear space that the numbers read as sitting *in* a cell, not
        # as filling it edge to edge.
        return widest + (0.18 if compact else 0.24)

    # Explicit inch-based budget — the axes are placed absolutely so that cell
    # size, and therefore label legibility, does not drift with row count.
    if row_label_in is None:
        # Fit the gutter to the longest row name rather than to a guess, so a
        # figure with short labels does not carry an inch of blank paper.
        row_label_in = _text_width_in(_row_names, fs_row) + 0.06
        if baseline_row:
            row_label_in = max(row_label_in,
                               _text_width_in(["baseline"], fs_base, style="italic") + 0.06)
    ylab_in = (0.16 if compact else 0.20) if row_axis_label else 0.0
    left_in = ylab_in + row_label_in
    # Height of the judge-name band: one line of type flat, or the vertical
    # reach of the longest name when it is slanted.
    if abs(rot) < 1.0:
        xlab_in = fs_col / 72.0 + 0.09
    else:
        xlab_in = (np.sin(np.radians(abs(rot))) * _text_width_in(_model_names, fs_col)
                   + 0.08)

    if cell_height_in is not None:
        cell_h = float(cell_height_in)
    base_row_in = base_h_in if baseline_row else 0.0
    grid_h_in = n_r * cell_h
    # Everything below the grid is optional; each part claims height only when
    # it is drawn, so dropping one closes the gap rather than leaving a hole.
    bot_pad = 0.04
    foot_in = (0.20 + 0.11 * footnote.count("\n")) if footnote else 0.0
    cbar_lab_in, cbar_in = (0.30, 0.10) if colorbar else (0.0, 0.0)
    mid_gap_in = 0.34 if colorbar else (0.16 if footnote else 0.0)
    xtitle_in = (0.16 if compact else 0.22) if col_axis_label else 0.0
    top_pad = 0.03
    grid_bottom_in = (bot_pad + foot_in + cbar_lab_in + cbar_in + mid_gap_in
                      + xtitle_in + xlab_in)

    # Stacked blocks share the full width and queue up vertically; side-by-side
    # blocks split the width and share one baseline of the figure.
    block_gap_in = (0.10 if compact else 0.30) if stacked else 0.0
    # Stacked blocks each carry their own judge names, so every block above the
    # bottom one needs that band inside its own footprint. The bottom block's is
    # already part of grid_bottom_in.
    xlab_extra = xlab_in if stacked else 0.0
    n_across = 1 if stacked else n_b
    avail_w = (width_in - left_in - 0.04 - gap_in * (n_across - 1)) / (n_across * n_c)
    # The block title is centred on its block, so a block narrower than its own
    # title hangs the chip off both sides.
    title_w_in = (_text_width_in([t for t, _ in blocks], fs_title, fontweight="bold")
                  + 2 * title_pad * fs_title / 72.0)
    if cell_width_in is not None:
        cell_w = min(cell_width_in, avail_w)
    elif compact:
        # Filling the width stretches a cell far past the five or six characters
        # it holds, and the slack reads as a gap between the numbers. Measure
        # what the cell actually has to fit and give it that much — but never so
        # little that the title no longer sits over its own block.
        cell_w = min(max(_fitted_cell_width_in(), title_w_in / n_c), avail_w)
    else:
        cell_w = avail_w
    block_w_in = n_c * cell_w
    title_band_in = max(title_in, fs_title / 72.0 * 1.3 + 2 * chip_pad_in)
    stride_in = (title_band_in + top_gap_in + base_row_in + grid_h_in
                 + block_gap_in + xlab_extra)
    n_down = n_b if stacked else 1
    fig_h = (grid_bottom_in + n_down * stride_in - block_gap_in - xlab_extra
             + top_pad)
    blocks_w_in = block_w_in * n_across + gap_in * (n_across - 1)
    # A fitted grid narrower than the page would otherwise trail blank paper on
    # the right, which is the slack we just removed from the cells.
    right_pad = max(0.04, (title_w_in - block_w_in) / 2 + 0.02)
    fig_w = min(width_in, left_in + blocks_w_in + right_pad)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    left = left_in / fig_w
    block_w = block_w_in / fig_w
    height = grid_h_in / fig_h

    for bi, (title, mats) in enumerate(blocks):
        if stacked:
            x0_in = left_in
            y0_in = grid_bottom_in + (n_b - 1 - bi) * stride_in
        else:
            x0_in = left_in + bi * (block_w_in + gap_in)
            y0_in = grid_bottom_in
        bottom = y0_in / fig_h
        # A stacked block stands on its own, so it repeats the row names; a
        # side-by-side one shares the gutter with the block to its left.
        show_rows = stacked or bi == 0

        ax = fig.add_axes([x0_in / fig_w, bottom, block_w, height])
        # pcolormesh rather than imshow: imshow embeds a tiny bitmap that the PDF
        # viewer would scale, this emits one vector rectangle per cell.
        ax.pcolormesh(np.arange(n_c + 1) - 0.5, np.arange(n_r + 1) - 0.5,
                      np.ma.masked_invalid(mats["delta"]),
                      cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
        ax.set_xlim(-0.5, n_c - 0.5)
        ax.set_ylim(n_r - 0.5, -0.5)   # row 0 at the top

        for i in range(n_r):
            for j in range(n_c):
                v = mats["delta"][i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=6, color="#999999")
                    continue
                norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                tc = "white" if norm_v < 0.25 or norm_v > 0.82 else "#1a1a1a"
                star = "?" if not mats["rel"][i, j] else ("*" if mats["sig"][i, j] else "")
                av = mats["abl"][i, j]
                show_abs = cell_absolute and np.isfinite(av)
                # With two lines the delta rides above centre and the absolute
                # sits under it; alone it stays centred.
                ax.text(j, i - cell_line_off if show_abs else i, f"{v:+.1f}{star}",
                        ha="center", va="center", fontsize=fs_delta, color=tc,
                        fontweight="bold")
                if show_abs:
                    # Italic so the absolute never reads as a second delta.
                    bv = mats["base"][i, j]
                    # With a per-row baseline there is no baseline strip to read
                    # the starting point from, so the cell carries both ends.
                    label = (f"({bv:.1f}→{av:.1f})"
                             if cell_before and np.isfinite(bv) else f"({av:.1f})")
                    ax.text(j, i + cell_line_off, label, ha="center", va="center",
                            fontsize=fs_abs, color=tc, alpha=0.72, style="italic")

        for x in range(n_c + 1):
            ax.axvline(x - 0.5, color="white", linewidth=grid_lw)
        for y in range(n_r + 1):
            ax.axhline(y - 0.5, color="white", linewidth=grid_lw)
        ax.set_yticks(range(n_r))
        if show_rows:
            ax.set_yticklabels(_row_names, fontsize=fs_row,
                               linespacing=1.15)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, pad=tick_pad)

        if baseline_row:
            bax = fig.add_axes([x0_in / fig_w, bottom + height,
                                block_w, base_row_in / fig_h])
            bax.set_xlim(-0.5, n_c - 0.5)
            bax.set_ylim(0, 1)
            bax.set_facecolor("#f4f6f8")
            for j in range(n_c):
                col = mats["base"][:, j]
                bv = next((x for x in col if np.isfinite(x)), np.nan)
                if np.isfinite(bv):
                    bax.text(j, 0.5, f"{bv:.1f}", ha="center", va="center",
                             fontsize=fs_base, color="#333333", fontweight="bold")
                bax.axvline(j - 0.5, color="white", linewidth=grid_lw)
            bax.set_yticks([])
            bax.set_xticks([])   # else the default ticks print into the grid
            for s in bax.spines.values():
                s.set_visible(False)
            if show_rows:
                # Right-aligned on the same edge as the row names, so the strip
                # reads as one more row of the grid.
                bax.text(-(tick_pad + 1.0) / 72.0 / block_w_in, 0.5, "baseline",
                         transform=bax.transAxes, ha="right", va="center",
                         fontsize=fs_base, color="#555555", style="italic")
            title_ax, title_y = bax, 1.0 + (top_gap_in + chip_pad_in) / base_row_in
        else:
            title_ax, title_y = ax, 1.0 + (top_gap_in + chip_pad_in) / grid_h_in

        # Judge names sit under the grid. The block title and the baseline strip
        # already own the top, and a third band of type there pushed the header
        # taller than the data it introduces.
        ax.set_xticks(range(n_c))
        ax.xaxis.set_ticks_position("bottom")
        # set_ticks_position turns the bottom labels back on, so the names are
        # set (or blanked) after it, not before.
        ax.set_xticklabels(
            _model_names, fontsize=fs_col, rotation=rot,
            ha="right" if abs(rot) >= 1.0 else "center",
            rotation_mode="anchor" if abs(rot) >= 1.0 else None)
        ax.tick_params(axis="x", length=0, pad=2.0)
        title_ax.text(0.5, title_y, title, transform=title_ax.transAxes,
                      ha="center", va="bottom", fontsize=fs_title, fontweight="bold",
                      bbox=dict(boxstyle=f"round,pad={title_pad}", fc="#eef1f5", ec="none"))

    if row_axis_label:
        # Centred on the whole stack of grids, not on one block, and sitting in
        # the width the budget already reserved for it.
        y_span_in = grid_h_in + (n_down - 1) * stride_in
        fig.text(0.4 * ylab_in / fig_w,
                 (grid_bottom_in + y_span_in / 2) / fig_h, row_axis_label,
                 rotation=90, ha="center", va="center", fontsize=fs_row + 0.6,
                 fontweight="bold", color="#333333")
    if col_axis_label:
        # Spans the blocks rather than one of them, in the band under the judge
        # names — whose height is measured above, so there is no hole to close.
        fig.text(left + blocks_w_in / 2 / fig_w,
                 (grid_bottom_in - xlab_in - xtitle_in / 2) / fig_h, col_axis_label,
                 ha="center", va="center", fontsize=fs_row + 0.6,
                 fontweight="bold", color="#333333")

    if colorbar:
        cax = fig.add_axes([left, (bot_pad + foot_in + cbar_lab_in) / fig_h,
                            blocks_w_in / fig_w, cbar_in / fig_h])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap),
            cax=cax, orientation="horizontal")
        cb.set_label(metric_label, fontsize=6.6, labelpad=1.5)
        cb.ax.tick_params(labelsize=5.8, length=2, pad=1)
        cb.outline.set_visible(False)
        # matplotlib rasterizes colourbar solids by default; keep the PDF vector.
        if cb.solids is not None:
            cb.solids.set_rasterized(False)

    if footnote:
        fig.text(left, bot_pad / fig_h, footnote, ha="left", va="bottom",
                 fontsize=5.4, color="#666666", style="italic", linespacing=1.5)

    return _save_figure(fig, out_path, formats=formats, dpi=dpi)


def _generate_two_track_forest(
    tracks: list,
    rows: list,
    models: list[str],
    metric: str,
    metric_label: str,
    out_path: "Path",
    model_labels: list[str] | None = None,
    width_in: float = 7.1,
    formats=None,
    dpi: int = 300,
    footnote: str | None = None,
) -> "list[Path] | None":
    """
    Dot-and-interval (forest) counterpart to `_generate_two_track_heatmap`.

    One line per (ablation, judge): dot = delta, whiskers = the bootstrap CI,
    filled = significant. Unlike the heatmap this distinguishes a tight null
    from an underpowered one, at the cost of roughly 1.4x the height.

    Requires chart data written with schema v2 or newer — v1 files carry no CI
    bounds. Returns None if no cell has an interval.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not tracks or not rows or not models:
        return None

    _model_names = list(model_labels) if model_labels else list(models)
    blocks = [(title, _collect_track_matrices(data, rows, metric, models, ti))
              for ti, (title, data) in enumerate(tracks)]
    have_ci = any(np.isfinite(m["ci_low"]).any() and np.isfinite(m["ci_high"]).any()
                  for _, m in blocks)
    if not have_ci:
        return None

    n_r, n_c, n_b = len(rows), len(models), len(blocks)
    # One slot per (ablation, judge), with a blank slot between ablation groups.
    slots, ylabels, group_bounds = [], [], []
    y = 0.0
    for i, (label, _) in enumerate(rows):
        start = y
        for j in range(n_c):
            slots.append((i, j, y))
            ylabels.append(_model_names[j])
            y += 1.0
        group_bounds.append((label, start, y - 1.0))
        y += 0.9
    total_h = y

    row_lab_in, grp_lab_in, gap_in = 0.62, 1.45, 0.35
    panel_w_in = (width_in - grp_lab_in - row_lab_in - gap_in * (n_b - 1)) / n_b
    fig_h = total_h * 0.115 + 1.0

    plt.rcParams.update({"font.family": "sans-serif", **_PDF_RCPARAMS})
    fig = plt.figure(figsize=(width_in, fig_h), facecolor="white")

    lows = np.concatenate([m["ci_low"][np.isfinite(m["ci_low"])] for _, m in blocks])
    highs = np.concatenate([m["ci_high"][np.isfinite(m["ci_high"])] for _, m in blocks])
    lo, hi = float(lows.min()), float(highs.max())
    pad = (hi - lo) * 0.05 or 1.0
    left0 = (grp_lab_in + row_lab_in) / width_in
    bottom, height = 0.62 / fig_h, (total_h * 0.115) / fig_h
    pw = panel_w_in / width_in

    for bi, (title, mats) in enumerate(blocks):
        ax = fig.add_axes([left0 + bi * (pw + gap_in / width_in), bottom, pw, height])
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(total_h - 0.5, -1.0)
        ax.axvline(0, color="#888888", linewidth=0.8, zorder=1)
        for gi, (_, gstart, gend) in enumerate(group_bounds):
            if gi % 2 == 0:
                ax.axhspan(gstart - 0.5, gend + 0.5, color="#f2f4f7", zorder=0)

        for i, j, yy in slots:
            d = mats["delta"][i, j]
            cl, ch = mats["ci_low"][i, j], mats["ci_high"][i, j]
            if not np.isfinite(d):
                continue
            sig = bool(mats["sig"][i, j])
            colour = "#98a2b3" if not sig else ("#c1121f" if d < 0 else "#2a7d54")
            if np.isfinite(cl) and np.isfinite(ch):
                ax.plot([cl, ch], [yy, yy], color=colour, linewidth=1.1,
                        solid_capstyle="round", zorder=2)
            ax.plot([d], [yy], marker="o", markersize=3.2, zorder=3,
                    markerfacecolor=colour if sig else "white",
                    markeredgecolor=colour, markeredgewidth=0.9)

        ax.set_yticks([s[2] for s in slots])
        ax.set_yticklabels(ylabels if bi == 0 else [], fontsize=5.2)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.tick_params(axis="x", labelsize=6.0, length=2, pad=1.5)
        ax.set_xlabel(metric_label, fontsize=6.4, labelpad=2)
        ax.grid(axis="x", color="#dde1e6", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c8ccd0")
        ax.set_title(title, fontsize=7.8, fontweight="bold", pad=5)

        if bi == 0:
            for glabel, gstart, gend in group_bounds:
                ax.text(-0.01 - row_lab_in / panel_w_in, (gstart + gend) / 2, glabel,
                        transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=6.8, fontweight="bold")

    fig.text(left0, 0.06 / fig_h,
             footnote or ("Filled dot = significant at 95%; hollow = not. Whiskers "
                          "are the corpus-level paired bootstrap CI (n=100,000)."),
             ha="left", va="bottom", fontsize=5.4, color="#666666", style="italic")

    return _save_figure(fig, out_path, formats=formats, dpi=dpi)
