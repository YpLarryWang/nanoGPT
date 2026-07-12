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
  `macro7` = mean of the seven task scores. `macro6` = mean of the six **excluding WSC**.
  WSC (Winograd coreference; only 554 train / ~104 dev examples) is too small and too hard to
  learn by fine-tuning at BabyLM scale, so every model collapses to the **majority class**
  (≈**63.46 % = 66/104**; minority-class F1 ≈ 0) — a near-constant that reflects the class prior,
  not model quality. `macro6` drops it, and is the GLUE analog of `reliable-4` on the zero-shot
  side (which drops the noisy `entity_tracking`). It can change conclusions: on `macro7` our 116M
  champion `bl100m-d512L32-do0.1-gate` trails Baseline-Strict (68.16 vs 68.86 — the gap is
  *entirely* WSC), but on `macro6` it edges ahead (68.94 vs 68.80).

## `source` column
- `ours(measured)` — run by us through the official pipeline. **Every row (both baselines
  included) is now our own measured number**, so all comparisons are apples-to-apples. Our
  measured baselines reproduce the official model cards closely (Strict blimp 74.73 vs 74.53;
  GLUE within ~1–2 pts) and additionally provide EWoK / WSC / avg5 / macro7 / macro6, which the cards omit.

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

## Scale-up experiments (`scale_up.csv`)
Separate table (2026-07-02) that scales the winner arch (RMS+SwiGLU 8/3+RoPE) **past the 33M
proof-of-concept**, holding arch fixed and sweeping depth/width/dropout on the 100M corpus
(`sampler=shuffle`, 10 epochs). Extra columns vs `zero_shot.csv`: `n_embd`, `n_layer`, `dropout`,
`sampler`, `best_val_loss`. Same metric definitions (avg5 = mean of the five accuracies; reading_*
= surprisal predictive power). Kept apart so the 33M architecture ablation stays clean.

Key rows / findings:
- **Winner `bl100m-d512L24-do0.1` (≈83M, deep+dropout): avg5 55.28 ± 0.63 (n=3 seeds) vs the 98M
  GPT-2 baseline's 54.71** — an 83M model matches/edges the 98M baseline at ~85% the params,
  winning BLiMP / EWoK / COMPS. The `-mean` row is the 3-seed average; `-s1337/-s1338/-s1339`
  are the individual seeds (single best seed hit avg5 56.04 — the seed pass corrected it down).
- **Dropout is scale-dependent:** at 33M dropout 0.1 cost 0.49 avg5, but at 83M `d512L24` do0.0→do0.1
  removes an end-of-training overfit (val gap 0.05→0) and lifts the reliable tasks (BLiMP +2.2).
- **Depth ≫ width** at fixed ~83M: `d512L24` (deep) > `d768L10` (wide), and the 97M `d768L12`
  regressed — width scaling plateaued.
- **entity_tracking is the noise source:** std 1.86 across the 3 winner seeds (range 21.2–25.6),
  which drives most of avg5's ±0.63; the reliable-4 (ex-entity) is tight and depth-monotonic.
