# Judge error annotation

Blind human annotation of the pairwise-novelty instances where a judge disagreed
with the gold label. Use it to sanity-check whether the judge's "mistakes" are
genuinely wrong, or whether the pair is ambiguous / the gold label is off.

## What it does

1. Reads a judge run's `scores.json` and the test-instances YAML it was scored
   against (path is auto-read from the run's `accuracy_report.txt`).
2. Selects the **mistake set**: instances where the judge's predicted winner
   (`elo_selected`) is not the gold winner (`expected_winners`). Instances whose
   papers appear in `benchmark_data/paper_blocklist.yaml` (title match, same rule
   as `analysis.filtering`) are dropped so they never surface for annotation.
3. Serves a small local web page that shows each pair's two ideas **blind** —
   left/right randomized, no scores, no gold, no "this was an error" hint — and
   asks which idea is more novel.
4. Writes every answer to a CSV, comparing human vs. judge vs. gold.

It can also export the mistake set as a standalone HTML report for sharing
(unblinded, read-only) — see [Export a shareable HTML report](#export-a-shareable-html-report).

## Run

```bash
export PYTHONPATH=$PWD/src
python -m novelty_eval.judge_error_annotation.annotate_judge_errors \
  --run-dir output/ablation_sweeps/20260518_110534/pairwise-vanilla-ai_current/pairwise-vanilla-ai_current-claude-opus-4-6 \
  --out-csv output/judge_error_annotations/opus-4-6.csv
```

Then open http://127.0.0.1:8000. Keys: `A`/`B` to pick, `T` for tie,
arrows to navigate. Answers autosave to the CSV on every click; re-running with
the same `--out-csv` resumes where you left off (already-answered items are
pre-filled and can be edited).

Options: `--test-instances PATH` to override the YAML, `--port`, `--host`,
`--seed` (controls the reproducible A/B blinding shuffle).

## Export a shareable HTML report

`--export-html` renders the same mistake set as a standalone HTML file and exits
without serving the UI. Unlike the annotation view this is **unblinded** — it is
meant for reading and sharing, not for annotating:

```bash
export PYTHONPATH=$PWD/src
python -m novelty_eval.judge_error_annotation.annotate_judge_errors \
  --run-dir output/ablation_sweeps/20260518_110534/pairwise-vanilla-ai_current/pairwise-vanilla-ai_current-claude-opus-4-6 \
  --export-html output/judge_error_annotations/judge_mistakes_opus-4-6.html
```

Each mistake becomes a card: instance id, area, and the gold winner and the
judge's pick as two tags in the header, then both ideas side by side (always
idea 0 then idea 1) with their titles, the ICLR contribution/rating, the full
idea text, and the review positive/negative signals in collapsible sections.
The idea columns are deliberately neutral — no colour-coding — so the header
tags are the single place the gold/predicted split is stated. The page has an
area filter and a free-text search, and is fully self-contained (inline CSS/JS,
no external assets), so it opens offline and can be emailed as a single file.

Swap `--run-dir` to export the same report for any other model in the sweep.

## CSV columns

| column | meaning |
|---|---|
| `problem_id` | instance id |
| `context` | research-area context shown to the annotator |
| `human_choice_ab` | what the annotator clicked: `A` / `B` / `tie` |
| `human_choice_idea` | decoded idea id (`0`/`1`) the annotator preferred, or `tie` |
| `judge_choice_idea` | judge's predicted winner (`elo_selected`) |
| `gold_idea` | ground-truth winner (`expected_winners[0]`) |
| `judge_novelty_winner` | argmax of the judge's novelty scores (or `tie`) |
| `human_correct` | `human_choice_idea == gold_idea` |
| `judge_correct` | `judge_choice_idea == gold_idea` (False across this set by construction) |
| `human_agrees_judge` | `human_choice_idea == judge_choice_idea` |
| `notes` | free-text notes |
| `shown_A_idea`, `shown_B_idea` | which idea was displayed as A / B (blinding record) |
| `annotated_at` | timestamp |

The headline questions the CSV answers: on the judge's mistakes, how often does
the human side with the **gold** (`human_correct`) vs. side with the **judge**
(`human_agrees_judge`) vs. call it a **tie**.

## Offline annotator batches (xlsx)

`prepare_annotation_batch.py` is the alternative to serving the UI: it ships each
annotator a single xlsx they fill in offline and send back, so annotators need
neither the repo nor a running server.

```bash
export PYTHONPATH=$PWD/src
python -m novelty_eval.judge_error_annotation.prepare_annotation_batch \
  --config src/novelty_eval/judge_error_annotation/annotation_batch.yaml
```

### Groups

A batch is assembled from named **groups**, each with its own `run_dir`, so one
batch can mix judges, runs and benchmarks. A group draws from one side of its
run's partition (`select: mistakes` or `controls`) and can require the judge to
have been decisive, via `min_novelty_margin` — the gap between the two novelty
scores on the 0-6 scale. The shipped config uses three:

| group | source | meaning |
|---|---|---|
| `mistakes` | the judge under study | the cases being investigated; ties included, since a tie is a failure |
| `controls` | same judge, `select: controls`, margin 6 | instances it got right by a 6-vs-0 call, so annotators have a baseline |
| `hard` | a *strong* judge on a different benchmark, margin 6 | pairs even opus-4-6 got confidently wrong |

Controls and hard examples both demand a **decisive** margin for the same
reason: a correct answer reached by a hair is a poor baseline, and an abstention
is not a strong judge failing. This constrains batch size — only 28 of the
sonnet-4-5 run's 61 correct instances are decisive.

Groups drawing on the same run must not overlap; the script refuses a batch that
would hand an annotator the same pair twice.

### Assignment

- `disjoint` (default) — everyone gets different instances, covering more of the
  pools, each of which must hold `num_annotators × per_annotator`. No measure of
  annotator agreement.
- `shared` — everyone annotates the same instances, with the same A/B blinding
  so answers line up column for column and agreement is a direct comparison.
  Only one set is drawn per group, so the pool requirement does not scale with
  the number of annotators. Row order still differs per annotator.

The script prints every group's pool size and refuses to under-fill a batch.

### The files

Each `<annotator>.xlsx` has an `instructions` sheet and an `annotation` sheet:

| `item_id` | `area` | `your_choice` | `notes` | `idea_A` | `idea_B` |
|---|---|---|---|---|---|

Alongside each xlsx the script writes `<annotator>.html`, a standalone page with
the same rows and the same blinding — one pair at a time, side by side,
answered with <kbd>A</kbd> / <kbd>B</kbd> / <kbd>U</kbd>. It opens offline with
nothing installed, keeps progress in the browser as you go, and exports a small
`answers_<annotator>.csv` to send back; loading that CSV into the page restores
the answers, so resuming never depends on browser storage surviving. The xlsx
remains the fallback for anyone who would rather use a spreadsheet, and both
routes produce the same answers keyed by `item_id`.

Rendering lives in `annotation_page.py`, apart from the selection and sampling
code, and it whitelists the four fields a page may show, so nothing that would
de-blind an annotator can reach the markup even if a caller passes richer rows.

Rows are identified by an opaque `item_id` rather than by `problem_id`. This is
required, not cosmetic: benchmarks number their instances from 1, so two sources
in one batch collide — the two files behind the shipped config both use ids
1-161 for entirely different pairs. The opaque id also stops group membership
being inferred from id ranges.

Blinding matches the UI: A/B order is randomized, no scores or gold labels
appear, and the groups are shuffled together with nothing marking which is
which.

`key.xlsx` holds the decoding — one row per (annotator, item) with `group`,
`source`, `problem_id`, `is_mistake`, `shown_A_idea`/`shown_B_idea`,
`gold_idea` and `judge_novelty_winner`. **Do not distribute it**;
`shown_A_idea`/`shown_B_idea` are what turn a returned `A`/`B` answer back into
an idea id, `gold_idea` is what makes it right or wrong, and `problem_id` only
means anything paired with its `source`.

### Annotating on a phone (Google Form)

Neither offline artifact works well on a phone: the xlsx wants a spreadsheet
app, and the page saves through `showSaveFilePicker`, which no mobile browser
has — there it falls back to re-downloading the answers CSV every few answers,
and its progress lives in browser storage that an emailed `file://` page may not
even have. `google_form_export.py` is the way round that, for one annotator or
all of them.

Google has no API for creating a form from outside, so the export is an Apps
Script that builds it:

```bash
export PYTHONPATH=$PWD/src
python -m novelty_eval.judge_error_annotation.google_form_export \
  output/judge_error_annotations/annotation-demo/tom.html
```

That writes `tom_form.gs` next to the page. Paste it into a new project at
script.google.com, run `buildForm()`, then run `verifyForm()`: it reads every
idea text back off the built form and compares it character for character with
the page, because Google is the only thing that can alter the text once the
script is written. The execution log prints the fill-in link to send and the
edit link to keep. The form is one page per pair, with a
required A / B / unclear question and a notes box, each tagged `[item NNN]` so
answers can be matched back. ### Layout

Submit sits at the end of the last page, so a form with one pair per page has to
be tapped all the way through before anything can be sent — which defeats
submitting part-way. By default every pair goes on one scrolling page, putting
Submit one scroll away at any moment. `--pairs-per-page N` splits the form
instead, at the cost of one Next tap per page; it also restores the progress
bar, which Forms only shows on a paged form.

### Partial answers

Only a **submitted** response reaches the responses CSV. An unsubmitted form is
a draft in the annotator's own Google account: the form owner cannot see it, and
no API can read it. So the questions are optional by default — the annotator
submits whatever they have whenever they stop, and reopens that response via the
"Edit your response" link on the confirmation page to carry on. Each submit
refreshes their row, and `--responses` merges the lot. Pass `--require-answers`
for the opposite trade: nobody can submit a half-finished batch, and nothing at
all appears until they finish.

Tell the annotator to keep that edit link — Forms shows it once, on the
confirmation page. Turning on "Send responders a copy of their response" in the
form's settings mails it to them instead, which needs email collection on.

The annotator should also be signed in to a Google account, so Forms keeps their
draft between sittings as a second line of defence.

Download the form's responses as CSV and convert them back into the same
`answers_<annotator>.csv` the page exports:

```bash
python -m novelty_eval.judge_error_annotation.google_form_export \
  output/judge_error_annotations/annotation-demo/tom.html --responses ~/Downloads/responses.csv
```

The file is written exactly as the page writes it — bare header, every value
quoted, LF endings — so a converted file and a returned one are the same
artifact. Later submissions win per item and a blank never overwrites an answer
already given, so a second sitting or an edited response can only add. Unanswered items
and `unclear` answers with no note are reported on stderr, the same two checks
the page makes. The resulting CSV also loads straight back into `tom.html`.

The export reads the rendered page rather than the batch config, because the
page is what fixes an annotator's row order and A/B blinding, and it carries
only the four whitelisted fields — so the form cannot show more than the page
did. Idea text is passed through untouched, markdown markers and all: it is
what two annotators are being compared on, so the export refuses to write a
script whose embedded text differs from the page's by a single character.
