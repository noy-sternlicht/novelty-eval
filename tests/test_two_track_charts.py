import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from src.novelty_eval.figures import unified_chart_data
from src.novelty_eval.figures.chart_two_track import (
    _common_rows,
    _row_key,
)
from src.novelty_eval.analysis.stats_one_pass import (
    compute_bootstrap_results_pairwise_all,
)
from src.novelty_eval.figures.viz import _generate_two_track_heatmap

MODELS = ["judge-a", "judge-b"]


def _panel(title, delta):
    """One panel where every model moves by `delta` from a 0.60 baseline."""
    bs = {}
    for m in MODELS:
        bs[(m, "pairwise_accuracy")] = {
            "significant": True, "reliable": True,
            "ci_low": delta - 0.02, "ci_high": delta + 0.02,
        }
    return {
        "title": title,
        "baseline_metrics": {m: {"pairwise_accuracy": 0.60} for m in MODELS},
        "ablation_metrics": {m: {"pairwise_accuracy": 0.60 + delta} for m in MODELS},
        "bootstrap_results": bs,
    }


def _track(plan_suffix):
    return {
        "setup": "pairwise",
        "all_models": list(MODELS),
        "rendered": ["vague_criterion", f"convert_to_plan_form_{plan_suffix}"],
        "panels": {
            "vague_criterion": _panel("Vague criterion", -0.05),
            f"convert_to_plan_form_{plan_suffix}": _panel("Plan form", 0.03),
        },
    }


# --- bootstrap now surfaces the interval, not just the booleans -------------

def test_pairwise_bootstrap_returns_ci_bounds():
    """CI bounds must be returned so charts need no re-bootstrap."""
    # (is_correct, is_tie, gt_winner) — gt must match positionally across the pair.
    base = [(1.0, 0.0, 0.0)] * 50 + [(0.0, 0.0, 0.0)] * 50
    abl = [(1.0, 0.0, 0.0)] * 80 + [(0.0, 0.0, 0.0)] * 20
    res = compute_bootstrap_results_pairwise_all(base, abl, n_resamples=500)
    entry = res["pairwise_accuracy"]
    assert {"significant", "reliable", "ci_low", "ci_high"} <= set(entry)
    assert entry["ci_low"] is not None and entry["ci_high"] is not None
    assert entry["ci_low"] <= entry["ci_high"]


def test_degenerate_bootstrap_ci_is_none_not_nan():
    """Non-finite bounds become None so the payload stays JSON-serialisable."""
    same = [(1.0, 0.0, 0.0)] * 40
    res = compute_bootstrap_results_pairwise_all(same, same, n_resamples=200)
    entry = res["pairwise_accuracy"]
    assert entry["ci_low"] is None and entry["ci_high"] is None
    json.dumps(entry)  # must not raise


# --- on-disk format ---------------------------------------------------------

def test_v1_data_still_loads(tmp_path):
    """Existing v1 files must keep working; they simply carry no interval."""
    p = tmp_path / "unified_pairwise.json"
    p.write_text(json.dumps({
        "schema_version": 1, "generated_at": "", "setup": "pairwise",
        "name_suffix": "", "variant": "unfiltered", "report_file": "r.txt",
        "all_models": MODELS, "metric_labels": {}, "available_metrics": {},
        "rendered": [], "panels": {},
    }))
    assert unified_chart_data.load(p)["schema_version"] == 1


def test_unknown_schema_version_is_rejected(tmp_path):
    p = tmp_path / "unified_pairwise.json"
    p.write_text(json.dumps({"schema_version": 99, "panels": {}}))
    with pytest.raises(ValueError, match="unsupported schema_version"):
        unified_chart_data.load(p)


def test_ci_bounds_survive_a_dump_load_round_trip(tmp_path):
    out = unified_chart_data.dump(
        tmp_path / "unified_pairwise.json",
        setup="pairwise", name_suffix="", report_file="r.txt",
        all_models=MODELS, metric_labels={"pairwise_accuracy": "Acc"},
        available_metrics={"pairwise_accuracy": "Acc"},
        rendered=["vague_criterion"],
        panels={"vague_criterion": dict(_panel("Vague", -0.05),
                                        canonical="vague_criterion")},
    )
    entry = unified_chart_data.load(out)["panels"]["vague_criterion"][
        "bootstrap_results"][("judge-a", "pairwise_accuracy")]
    assert entry["ci_low"] == pytest.approx(-0.07)
    assert entry["ci_high"] == pytest.approx(-0.03)


# --- row matching across differently-named tracks ---------------------------

def test_row_key_strips_track_and_setup_tokens():
    """One `rows:` entry must cover both setups, so both tokens come off."""
    from src.novelty_eval.figures.chart_two_track import (
        DEFAULT_TRACK_SUFFIXES as S,
    )
    assert _row_key("convert_to_plan_form_pairwise_human", S) == "convert_to_plan_form"
    assert _row_key("convert_to_plan_form_pointwise_vanilla", S) == "convert_to_plan_form"
    assert _row_key("vague_criterion", S) == "vague_criterion"
    # A prefix that happens to be a token must not be touched.
    assert _row_key("pointwise_retrieval", S) == "pointwise_retrieval"
    # A name that is nothing but a token must survive intact.
    assert _row_key("pairwise", S) == "pairwise"


def test_setup_specific_row_name_still_matches_the_other_setup():
    """A pairwise-named row must resolve against pointwise data, not vanish."""
    pointwise_tracks = [_track("pointwise_human"), _track("pointwise_vanilla")]
    rows = _common_rows(pointwise_tracks, ["convert_to_plan_form_pairwise_human"], {})
    assert len(rows) == 1, "setup-specific name should not drop the row"
    assert rows[0][1] == ("convert_to_plan_form_pointwise_human",
                          "convert_to_plan_form_pointwise_vanilla")


def test_rows_match_across_per_track_ablation_names():
    """`..._human` in one track must line up with `..._vanilla` in the other."""
    rows = _common_rows([_track("pairwise_human"), _track("pairwise_vanilla")],
                        None, {})
    labels = [r[0] for r in rows]
    assert "Plan form" in labels, "per-track variants should share one row"
    plan = next(r for r in rows if r[0] == "Plan form")
    assert plan[1] == ("convert_to_plan_form_pairwise_human",
                       "convert_to_plan_form_pairwise_vanilla")


def test_row_absent_from_one_track_still_renders():
    """A row only one track has should survive, as an empty cell elsewhere."""
    other = _track("pairwise_vanilla")
    del other["panels"]["convert_to_plan_form_pairwise_vanilla"]
    rows = _common_rows([_track("pairwise_human"), other],
                        ["convert_to_plan_form_pairwise_human"], {})
    assert len(rows) == 1
    assert rows[0][1] == ("convert_to_plan_form_pairwise_human", None)


def test_explicit_row_labels_win_over_data_titles():
    rows = _common_rows([_track("pairwise_human"), _track("pairwise_vanilla")],
                        ["vague_criterion"], {"vague_criterion": "Custom"})
    assert rows[0][0] == "Custom"


# --- rendering --------------------------------------------------------------

def _tracks_and_rows():
    tracks = [("Human-only", _track("pairwise_human")),
              ("Human + generated", _track("pairwise_vanilla"))]
    rows = _common_rows([d for _, d in tracks], None, {})
    return tracks, rows


def test_heatmap_renders_pdf_and_png(tmp_path):
    tracks, rows = _tracks_and_rows()
    written = _generate_two_track_heatmap(
        tracks=tracks, rows=rows, models=MODELS, metric="pairwise_accuracy",
        metric_label="delta", out_path=tmp_path / "fig.pdf", formats=["png"])
    assert written is not None
    assert {p.suffix for p in written} == {".pdf", ".png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_heatmap_declines_when_no_finite_data(tmp_path):
    tracks, rows = _tracks_and_rows()
    assert _generate_two_track_heatmap(
        tracks=tracks, rows=rows, models=MODELS, metric="does_not_exist",
        metric_label="delta", out_path=tmp_path / "fig.pdf") is None


def test_stacked_compact_heatmap_is_narrow_and_tall(tmp_path):
    """The one-column layout must fit its width and grow downward instead."""
    from PIL import Image

    tracks, rows = _tracks_and_rows()
    kwargs = dict(tracks=tracks, rows=rows, models=MODELS,
                  metric="pairwise_accuracy", metric_label="delta")
    wide = _generate_two_track_heatmap(out_path=tmp_path / "wide.png", **kwargs)[0]
    onecol = _generate_two_track_heatmap(
        out_path=tmp_path / "onecol.png", width_in=3.3, stacked=True, compact=True,
        **kwargs)[0]
    w_wide, h_wide = Image.open(wide).size
    w_one, h_one = Image.open(onecol).size
    assert w_one < w_wide / 2
    assert h_one > h_wide


def test_compact_heatmap_fits_the_cells_to_their_numbers(tmp_path):
    """--compact buys back height, and width the stretched cells were wasting."""
    from PIL import Image

    tracks, rows = _tracks_and_rows()
    kwargs = dict(tracks=tracks, rows=rows, models=MODELS,
                  metric="pairwise_accuracy", metric_label="delta")
    default = _generate_two_track_heatmap(out_path=tmp_path / "d.png", **kwargs)[0]
    compact = _generate_two_track_heatmap(out_path=tmp_path / "c.png", compact=True,
                                          **kwargs)[0]
    (w_d, h_d), (w_c, h_c) = Image.open(default).size, Image.open(compact).size
    assert h_c < h_d
    # `width_in` is a ceiling once the cells are fitted, not a target.
    assert w_c < w_d


def test_explicit_cell_size_overrides_the_fit(tmp_path):
    """--cell-width/--cell-height have to win over both fit and preset."""
    from PIL import Image

    tracks, rows = _tracks_and_rows()
    kwargs = dict(tracks=tracks, rows=rows, models=MODELS, compact=True,
                  metric="pairwise_accuracy", metric_label="delta")
    small = _generate_two_track_heatmap(out_path=tmp_path / "s.png",
                                        cell_width_in=0.5, cell_height_in=0.3,
                                        **kwargs)[0]
    big = _generate_two_track_heatmap(out_path=tmp_path / "b.png",
                                      cell_width_in=0.9, cell_height_in=0.5,
                                      **kwargs)[0]
    (w_s, h_s), (w_b, h_b) = Image.open(small).size, Image.open(big).size
    assert w_b > w_s and h_b > h_s
    # 2 rows down, so the extra height lands in the grid and nowhere else.
    assert h_b - h_s == pytest.approx(2 * 0.2 * 300, abs=2)
    # 4 cells across, less the title overhang the wider blocks no longer need.
    assert 0 < w_b - w_s <= 4 * 0.4 * 300 + 2
