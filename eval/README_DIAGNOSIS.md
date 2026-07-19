# L32 diagnosis execution

This workflow implements `DIAGNOSIS_DESIGN_v2.md`. It performs no training.
All model forward passes run on JetStream/Vast; local work is limited to
artifact audit, collation, statistics, and figures.

## 1. Deploy and preflight

Commit and push this branch, pull the same commit on both GPU servers, then run
the read-only local preflight:

```bash
EXPECTED_SHA=<commit> bash eval/diag_remote_preflight.sh
```

The preflight requires the s1337 pair on JetStream, the s1338/s1339 pairs on
Vast, and all six selected checkpoints in the T9 backup. It also rejects a
remote Git SHA mismatch.

## 2. Pick A on each GPU server

Run `diag_dev_series.py` once per assigned run. The native training environment
loads checkpoints for fixed-dev NLL; the separately pinned BabyLM eval
environment converts checkpoints and evaluates full BLiMP.

```bash
<driver-python> eval/diag_dev_series.py \
  --run-dir out-babylm/<run> \
  --data-dir data/babylm_officialdev \
  --eval-root <babylm-eval>/strict \
  --tokenizer data/babylm_officialdev/tokenizer/bpe-16000.json \
  --output-root <diagnosis-raw>/series \
  --cache-root <diagnosis-cache>/hf \
  --native-python <nanogpt-env>/bin/python \
  --eval-python <babylm-eval-env>/bin/python \
  --resume
```

Assignments are fixed:

- JetStream: baseline and AttnRes s1337;
- Vast: baseline and AttnRes for s1338 and s1339.

The driver evaluates 1M/5M/10M/20M/50M/final fixed-dev NLL plus position
statistics. It runs full BLiMP for the five intermediate checkpoints and
reuses an already complete final BLiMP tree.

## 3. Pick B on final AttnRes checkpoints

Run the three intervention modes for each assigned AttnRes run:

```bash
RESUME=1 bash eval/diag_masked_zero_shot.sh <run> old
RESUME=1 bash eval/diag_masked_zero_shot.sh <run> embed
RESUME=1 bash eval/diag_masked_zero_shot.sh <run> random_count_matched
```

The script evaluates full BLiMP, COMPS, and entity tracking. Both evaluator tie
breaking and the random control use the frozen seed `20260718`. `none` is also
supported for the one-checkpoint parity smoke, but final unmasked task trees are
otherwise reused. Parity requires byte-identical predictions and identical
temperature/average report headlines; a granular report can differ when the
legacy unseeded evaluator randomly chose between duplicate candidate strings.

Vast storage is not persistent. Sync each completed series/result tree to the
local backup immediately; do not wait for the full grid.

## 4. Local collation and figures

After syncing all raw artifacts:

```bash
python eval/diag_parse_results.py \
  --series-root <synced-series-root> \
  --eval-results-root <synced-jetstream-results> \
  --eval-results-root <synced-vast-results> \
  --output-dir eval/results/diag

MPLCONFIGDIR=/tmp/nanogpt-mpl-cache python eval/diag_make_figures.py \
  --input-dir eval/results/diag \
  --output-dir eval/results/diag \
  --figure-dir proposal/figures
```

The parser is intentionally strict: missing report/prediction pairs, missing
mask modes, duplicate trees, non-six-point plans, or a BLiMP report without all
13 native terms are fatal. Use `--allow-incomplete` only for an explicit
mid-run audit, never for final figures.
