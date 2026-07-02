# BabyLM-2026 strict-track — evaluation scoreboard (CSV)

Kept as CSV so it imports straight into Google Sheets
(**File → Import → Insert new sheet(s)**). Import each file as its own tab.

## Comparison frame (important)
- `bl100m-*` = our nanoGPT ablations, **33M params**, trained on the **100M-word** strict corpus.
- `bl10m-*`  = the same architecture trained on the **10M-word** strict-small corpus.
- Baselines  = official BabyLM GPT-2 (**98M params** — 3× larger than ours;
  `strict` = 100M words, `strict-small` = 10M words).

So the fair rows to compare are: **bl100m-\*  ↔  Baseline-Strict** and **bl10m-\*  ↔  Baseline-Strict-Small**.
Our headline story: a 33M model matching/apporaching a 98M GPT-2 at ~1/3 the parameters.

## Files
- `zero_shot.csv` — full-data zero-shot. Columns BLiMP / BLiMP-supplement / EWoK /
  entity_tracking / COMPS are **accuracy (%)**; `avg5` = mean of those five.
  `reading_eye` / `reading_selfpaced` = surprisal **predictive power** over human reading
  times: % of residual RT variance explained, ((R²_model−R²_base)/(1−R²_base))×100, averaged
  over the eye-tracking measures / the self-paced measure (NOT a correlation; not part of `avg5`).
  `eval_tok` = which tokenizer produced each row's numbers — see **Tokenizer fix** below.
- `glue.csv` — GLUE fine-tuning, reported with the **official per-task metric**:
  boolq / multirc / rte / wsc / mnli = **accuracy**; mrpc / qqp = **F1** (all %).
  `macro7` = mean of the seven task scores.

## `source` column
- `ours(measured)` — run by us through the official pipeline. **Every row (both baselines
  included) is now our own measured number**, so all comparisons are apples-to-apples. Our
  measured baselines reproduce the official model cards closely (Strict blimp 74.73 vs 74.53;
  GLUE within ~1–2 pts) and additionally provide EWoK / WSC / avg5 / macro7, which the cards omit.

## Tokenizer fix (reading scores) — and the `eval_tok` column
`reading_*` read ~0 for every nanoGPT variant (winner 0.24 / 0.10) while the GPT-2 baselines
were fine (10.54 / 3.32) — a converter bug, not a model property. `convert_nanogpt_to_hf.py`
wrapped our byte-level BPE without `add_prefix_space`, so the eval tokenizer mapped a bare word
to the no-space subword (`dog`) instead of the space-prefixed `Ġdog` the model predicts mid-text;
the reading harness scores `P(target[0])`, so it read the wrong token. Setting
`add_prefix_space=True` (matches training + the GPT-2 convention) fixed it — winner verified
**0.24/0.10 → 8.70/4.47**. **`reading_*` is now post-fix for every row.** The `eval_tok` column
marks the provenance of the **accuracy** (non-reading) columns per row:

| `eval_tok` | meaning | rows |
|---|---|---|
| `all-fixed` | every metric under the corrected tokenizer | **all 9 ablations** (winner + 8 others + bl10m) |
| `native` | official GPT-2 baseline tokenizer; never affected | both baselines |

All 9 ablation rows were re-run under the fix on 2026-07-02 (the 5 non-reading tasks; `reading_*`
was already post-fix and preserved), so the accuracy columns are now uniform `all-fixed`. The fix's
systematic effect — **supplement ↑, entity_tracking ↓** (≤~2 pts) — reshuffled the mid-table and
narrowed the top two to a near-tie: winner `avg5` 54.20 vs `rms-mlp-learned` 54.18 (a 0.57 gap
pre-fix). Biggest rank movers: `ln-swiglu-learned` ↑3, `ln-mlp-rope` ↓3, `rms-swiglu4-rope` ↓3.

## Coverage note
- Zero-shot (`zero_shot.csv`): all 9 bl100m variants, our bl10m winner, and both baselines.
- GLUE (`glue.csv`): our two 100M models (winner + rms-mlp-learned) and **both baselines**.
  Our **10M** model has zero-shot only — no GLUE run yet — so there is no `bl10m` row in
  `glue.csv` (a 10M-track GLUE head-to-head would need ~2.5 h to fine-tune bl10m-rms-swiglu-rope).

## Winner GLUE secondary metrics (reference, not in the CSV)
`bl100m-rms-swiglu-rope`: boolq f1 78.91 / mcc 31.02; multirc f1 44.03; rte f1 54.55;
wsc f1 24.0; mrpc acc 72.06; qqp acc 76.80.

## Bottom line
Across our own axes the winner **`bl100m-rms-swiglu-rope`** (RMS-norm + SwiGLU 8/3 + RoPE)
leads on all three: lowest val loss, highest zero-shot `avg5` (54.20), and highest GLUE
`macro7` (65.92) — despite being 1/3 the size of the GPT-2 baseline.
