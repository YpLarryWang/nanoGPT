# BabyLM 2026 Project Memory

> Last updated: 2026-07-17. This file is the durable handoff for the current
> submission phase. Read it together with `AGENTS.md`; when the two disagree,
> prefer newer measured data and verify the relevant CSV/raw artifact before
> acting. Some historical prose in `AGENTS.md` and `eval/results/README.md`
> predates the final 116M hybrid runs.

## 1. What this project is optimizing for

This nanoGPT fork is being used for the BabyLM 2026 strict (100M-word corpus)
and strict-small (10M-word corpus) tracks. We are in deadline mode:

- prioritize a defensible, reproducible submission over new speculative work;
- keep presentation numbers traceable to raw prediction/evaluation artifacts;
- distinguish internal research metrics from leaderboard metrics;
- do not let a convenient aggregate silently mix incompatible model modes;
- preserve evidence even when the result is weak or zero.

The current architecture family is approximately 116M parameters:
`d512L32`, RMSNorm + SwiGLU (8/3) + RoPE + gated attention, dropout 0.1,
16k vocabulary, context length 512, AdamW, shuffle sampler.

### Locked data protocol from 2026-07-16

All new experiments use the organizer-aligned official-development protocol:

```text
complete cleaned official train -> train.bin
complete cleaned official dev   -> val.bin
tokenizer input                  -> cleaned train only
```

The previous protocol carved a contiguous ~1% tail from every cleaned official
training source. It remains valid historical evidence, but it is now legacy and
must not be used for new runs. Preserve its existing data directories,
checkpoints, schedules, and result rows unchanged.

New data lives in parallel directories:

```text
data/babylm_officialdev/          # strict-small, 10M official train
data/babylm_100m_officialdev/     # strict, 100M official train
```

Both use the pinned official development release
`BabyLM-community/BabyLM-dev@169f42e32d0aaf65ec6b91d55bafad27a3afc729`.
The six raw dev files are named `<source>.dev` (no `.txt` suffix). The same dev
text is used for both tracks, but each track gets its own `val.bin` because its
tokenizer is trained independently on that track's full cleaned train split.

Protocol consequences are hard requirements:

- the new pipeline has no dependency on the legacy tail `split.py`;
- retrain every tokenizer from the full cleaned official train and record its
  train-only input fingerprints;
- all new formal run names include `-offdev`, and their `dataset` is
  `babylm_officialdev` or `babylm_100m_officialdev`;
- never reuse the old 466/4740 iteration counts for offdev runs; derive training
  length and all AoA checkpoint schedules from the newly built `train.bin` and
  its measured word/token exposure;
- before formal training, save raw/clean word counts, source revisions and
  SHA256 fingerprints, a train/dev overlap audit, and a short no-W&B smoke test;
- do not change the cleaning policy while changing the split protocol, so the
  protocol transition remains interpretable.

P0 construction completed on Jetstream at commit
`f55dba2daf28c48164037a2408415d58b0984b28`:

| dataset | raw train words | clean train words | train BPE tokens | official-dev BPE tokens | derived 10-epoch max_iters |
|---|---:|---:|---:|---:|---:|
| `babylm_officialdev` | 10,000,000 | 9,031,000 | 12,342,269 | 13,239,606 | 471 |
| `babylm_100m_officialdev` | 100,000,000 | 91,013,313 | 125,741,676 | 13,118,445 | 4,797 |

The max-iteration values are derived from the measured train bins at
32 batch x 16 gradient accumulation x 512 context = 262,144 tokens/update.
They are inputs to future schedule construction, not replacements to paste into
legacy runners.

Final fingerprints:

| dataset | tokenizer SHA256 | train.bin SHA256 | val.bin SHA256 |
|---|---|---|---|
| 10M offdev | `8c877bb7243db5669f68a9bbbf0c46ca56fb02edd2d43520aac978aa1f35a873` | `63871a140cb32e170d848ca8f13be39801ddf6d50809bb4d997510e99a7eeb10` | `c7e638f3e7c5afbce3503e09d9ddd46565a1a43d27fb46761c8216b747f05007` |
| 100M offdev | `f13720328807e761dc92192111d89ece0119987875f890f3665eba477b5d727c` | `8b3f3e41b28ab3cfde1fe8f1c095c662ece3332f081d1e70698a279d0e1a6853` | `d078706863afbcf3b35a6d1e7f5f571eab99c7ac52bda8fbe2c6a0fbe4ad2fa8` |

The official releases themselves contain exact cleaned-line overlap between
train and dev. The global audit reports 35,425 unique exact / 225 long-line
overlaps at 10M and 116,003 exact / 5,360 long-line overlaps at 100M. "Long"
means at least five whitespace words and 40 characters. This is source-data
overlap, not pipeline leakage: downloads are independently pinned and hashed,
and both tokenizer manifests verify exactly six `clean/train` inputs with no
dev input. Preserve and disclose the audit rather than claiming zero overlap.

Both datasets passed a two-update GPU smoke with shuffle sampling and official
dev validation (`wandb_log=False`, logs/checkpoints under `/tmp`): final smoke
validation loss was 9.6844 for 10M and 9.6786 for 100M. These losses only prove
the data/training path executes; they are not experimental results.

The formal B32/GA16 AoA schedules are committed for sampler seeds 1337/1338/1339.
Every new formal run must pass its seed-specific dual schedule; the word ladder
depends on shuffle order even though the token ladder does not. At seed 1337:

| track | final words | final tokens | word labels | token labels | unique checkpoints |
|---|---:|---:|---:|---:|---:|
| 10M offdev | 90,343,884 | 123,469,824 | 19 | 20 | 37 |
| 100M offdev | 910,196,864 | 1,257,504,768 | 28 | 31 | 57 |

Use `run_babylm_offdev_aoa.sh`; do not repoint the legacy AoA runners or their
old schedule files. The offdev runner includes `-offdev` in W&B/output names,
uses max iterations 471/4797, and fails closed if the seed-specific schedule is
missing or its parameters disagree with the run. `train.py` also rejects every
formal offdev W&B run that omits `checkpoint_schedule`, so alternate runners
cannot silently skip AoA coverage.

## 2. Current decision snapshot

### 100M

Current primary checkpoint family:

```text
bl100m-d512L32-do0.1-gate-hyb15of16-aoaw27-aoat31-u56-b32ga16
```

The manifest-best checkpoint is the final checkpoint:

```text
ckpt_best-w0899M-i004740.pt
iter_num=4740, words_seen=899,335,091, tokens_seen=1,242,562,560
```

The same checkpoint has been evaluated in two modes. They are separate result
rows and must remain separate:

| mode | reliable4 | local macro7 | leaderboard `(Super)GLUE` | NLP avg | AoA | overall |
|---|---:|---:|---:|---:|---:|---:|
| causal | 64.05 | 68.76 | 69.10 | 54.70 | 0.00 | 43.33 |
| bidirectional/MNTP | 62.35 | 69.96 | 69.99 | 53.91 | missing | missing |

Important: the old “dual-ladder” headline (causal zero-shot + bidirectional
GLUE) is useful mechanistic evidence, but it is **not one legal leaderboard
row** unless the submission rules explicitly allow mode switching. Never quote
the mixed ~54.8 NLP aggregate as if it came from one evaluation mode. Show both
rows and verify the submission rule before packaging.

The 100M fallback/reference is:

```text
bl100m-d512L32-do0.1-gate-aoa28-b32ga16
```

Its corrected values include Entity 18.14, GlobalPIQA 39.65,
`superglue_ld=68.18`, and `nlp_avg=54.38`.

### 10M

The preregistered decision used the internal `macro7` threshold 63.7:

- hybrid s1338 causal FT: `macro7=63.09`;
- gate-AoA19 B32/GA16 fallback: `macro7=63.71`.

Therefore the current 10M fallback/submission candidate is:

```text
bl10m-d512L32-do0.1-gate-aoa19-b32ga16
```

Key fallback values: reliable4 56.21, Entity 19.96, GlobalPIQA 29.20,
local macro7 63.71, leaderboard `(Super)GLUE` 63.67, NLP average 48.24.

The hybrid s1338 rows remain useful evidence:

| mode | reliable4 | local macro7 | leaderboard `(Super)GLUE` | NLP avg | AoA |
|---|---:|---:|---:|---:|---:|
| causal | 56.80 | 63.09 | 63.04 | 48.60 | 0.00 (`min_context=20`) |
| bidirectional/MNTP | 57.05 | 63.07 | 62.97 | 48.98 | missing |

The 10M macro7 gap is concentrated in RTE, not uniform across tasks. When
presenting it, always state the subtraction direction explicitly:

```text
hybrid s1338 causal minus gate-AoA19 fallback
```

## 3. Metric definitions: never blur these again

### Internal zero-shot metrics

```text
avg5 = mean(BLiMP, Supplement, EWoK, Entity Tracking, COMPS)
reliable4 = mean(BLiMP, Supplement, EWoK, COMPS)
```

`reliable4` drops Entity Tracking because Entity was historically noisy and
was also affected by an evaluation-layout change. Use it for internal model
selection narratives, but do not call it a leaderboard aggregate.

### Internal GLUE metrics

`macro7` keeps the original per-task metric convention:

- accuracy: BoolQ, MultiRC, RTE, WSC, MNLI;
- F1: MRPC and QQP.

`macro6` is the same convention excluding WSC. These are research metrics,
not the public leaderboard `(Super)GLUE` column.

### Leaderboard `(Super)GLUE`

`superglue_ld` is the equal-weight mean of **accuracy for all seven tasks**:

```text
BoolQ, MultiRC, RTE, WSC, MRPC, QQP, MNLI
```

Keep both `macro7` and `superglue_ld`. Never overwrite one with the other.

### GlobalPIQA and leaderboard aggregates

```text
GlobalPIQA = (parallel + nonparallel) / 2
Reading = (reading_eye + reading_selfpaced) / 2

nlp_avg = mean(
  BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, superglue_ld
)
human_like = mean(Reading, AoA)
overall_avg = mean(the seven NLP components, Reading, AoA)
```

The two GlobalPIQA splits have equal weight even though they contain 103 and
100 English examples. Low parallel scores are real inputs to the official
aggregate; do not omit the split to avoid a weak number.

The public leaderboard treats a missing submitted task as zero. Locally,
`all_runs.csv` deliberately leaves derived aggregates blank until all inputs
exist. This prevents “not measured” from becoming indistinguishable from a
genuine measured `0.00`.

## 4. Evaluation corrections that must remain permanent

### Entity Tracking

Legacy predictions contained 9,483 examples, including options containing
`nothing`. The current official layout keeps 6,780 examples and removes 2,703.

Known correction:

```text
100M gate-AoA28 Entity: 29.41 -> 18.14
```

Entity must be recomputed from `predictions.json` plus the official JSONL
filter. Do not copy the number from a legacy `best_temperature_report.txt`.
The recomputation lives in `eval/audit_entity_tracking.py` and is invoked by
the structured backup importer.

### Tokenizer / reading

The HF wrapper must use the corrected leading-space behavior
(`add_prefix_space=True`). The old wrapper made reading scores near zero by
scoring the wrong first token. Do not compare pre-fix and post-fix reading
numbers.

### AoA

`n_words` in `aoa_score.json` is **not** the number of input target words. It
is the number remaining after CDI matching, child-curve fitting, and model
learning-curve fitting.

Current AoA stage rows:

| model | AoA | fitted words | min_context | checkpoints |
|---|---:|---:|---:|---:|
| 100M u56 causal | 0.00 | 10 | 0 | 27/27 |
| 10M hybrid s1337 | 0.00 | 231 | 20 | 18/18 |
| 10M hybrid s1338 | 0.00 | 226 | 20 | 18/18 |

Zero AoA is common on the leaderboard and should remain a measured zero.
However, submission runs should use `min_context=0`. The two 10M AoA results
must be rerun with `min_context=0` before using them as submission artifacts.

`eval/run_local_aoa.py` now records `min_context` in `surprisal.json` metadata.
`eval/sync_aoa_results.py` writes it to `aoa.csv`, rejects new results where it
is missing, and prefers a complete `min_context=0` rerun over older nonzero
results. The three older results are covered by an explicit audited legacy
mapping in that script.

## 5. Result data flow: raw evidence -> stage tables -> all_runs

`all_runs.csv` is generated output. It is not the place to calculate or repair
metrics by hand.

```text
raw BabyLM eval directories / AoA JSON
    -> eval/import_eval_backups.py or a focused sync script
    -> stage CSVs
    -> eval/build_all_runs.py
    -> eval/results/all_runs.csv
    -> eval/validate_results.py
```

Stage/source-of-truth files:

- `eval/results/zero_shot.csv`
- `eval/results/scale_up.csv`
- `eval/results/glue.csv`
- `eval/results/aoa.csv`
- `eval/results/training_metadata.csv`

Generated union:

- `eval/results/all_runs.csv`

Main batch command:

```bash
python eval/import_eval_backups.py \
  /Users/yupeiwang/Documents/checkpoint-backups/babylm-2026
```

This importer:

- merges zero-shot tasks archived at different times;
- requires duplicate raw reports to agree;
- refuses partial seven-task GLUE runs;
- recomputes filtered Entity from saved predictions;
- imports MRPC/QQP accuracy as well as F1;
- imports complete AoA and records coverage/configuration;
- rebuilds and validates `all_runs.csv` once.

Focused commands:

```bash
python eval/sync_eval_results.py MODEL --glue
python eval/sync_eval_results.py MODEL --full --backend causal
python eval/sync_eval_results.py MODEL --fast --all-revisions --backend causal
python eval/sync_aoa_results.py /path/to/backup/root
python eval/build_all_runs.py
python eval/validate_results.py
python -m unittest discover -s eval/tests -p 'test_*.py'
```

Current validation baseline: 75 unified runs and 18 unit tests passing.

## 6. Important directories and machines

### Local

```text
repo:
/Users/yupeiwang/Documents/experiment/nanoGPT

raw checkpoint/eval backups:
/Users/yupeiwang/Documents/checkpoint-backups/babylm-2026

2026-07-15 Jetstream drain:
/Users/yupeiwang/Documents/checkpoint-backups/babylm-2026/2026-07-15/jetstream-eval-results

official Entity JSONL backup:
/Users/yupeiwang/Documents/checkpoint-backups/babylm-2026/2026-07-15/evaluation-data/entity_tracking

current slides:
presentations/0714/babylm_submission_update_0714_v4.pptx

older slide/template source:
presentations/0706/babylm_findings.tex
```

`presentations/` is gitignored. Do not assume a slide edit is preserved by a
code commit.

### Jetstream

SSH alias: `jetstream-fv-agop`

```text
nanoGPT repo:
/media/volume/yupei-data/repo/nanoGPT

BabyLM eval repo:
/media/volume/yupei-data/repo/babylm-eval/strict

training environment:
/media/volume/yupei-data/envs/nanogpt

evaluation environment:
/media/volume/yupei-data/envs/babylm-eval

100M AoA raw result:
/media/volume/yupei-data/aoa-results/bl100m-hyb-full-contexts/
  bl100m-d512L32-do0.1-gate-hyb15of16-aoaw27-aoat31-u56-b32ga16
```

Training and evaluation environments are intentionally separate. The eval
environment pins older Transformers dependencies; mixing them causes subtle
failures.

## 7. Engineering discipline (hard rules)

1. **Never hand-calculate into `all_runs.csv`.** Fix/import the stage source,
   rebuild, and validate.
2. **Raw artifacts come first.** Back up predictions, per-task reports,
   `aoa_score.json`, `surprisal.json`, and manifests before cleaning a remote
   machine.
3. **One row means one model mode.** Causal and bidirectional/MNTP evaluations
   get separate rows and names. Do not splice their best columns together.
4. **Missing and zero are different.** Blank means not measured; `0.00` is a
   measured result and must survive parsing.
5. **Fail closed.** Partial GLUE, conflicting duplicate backups, incomplete AoA,
   missing AoA configuration, or ambiguous result roots should raise errors.
6. **Record denominator/configuration.** Keep prediction counts, checkpoint
   coverage, `n_words`, `min_context`, backend, revision, and source path.
7. **Every non-default training flag belongs in the run name.** The run name
   propagates to W&B, `out_dir`, manifests, HF artifacts, and result rows.
8. **Formal runs log to W&B.** Never rely on undocumented defaults or terminal
   history as the only provenance.
9. **Deploy code through git only.** Commit -> push -> pull -> verify matching
   SHAs. `rsync` is for data/results, never for code deployment.
10. **Use manifest roles, not filename intuition.** Resolve `roles.best` and
    verify `iter_num`, words, tokens, and SHA before evaluation.
11. **Inspect every per-task result.** BabyLM GLUE can silently leave a failed
    task; the presence of a parent directory is not proof of completeness.
12. **Preserve the user's dirty worktree.** Do not revert or rewrite unrelated
    local changes. `data/babylm/bpe.py` is deliberately unfinished/untracked.
13. **CSV changes are reviewed separately.** After an eval/import, show the
    exact affected rows and wait for user approval before committing.
14. **Slides quote the CSV, not memory.** If a metric definition changes,
    update the table and explanatory language together.
15. **Do not overinterpret tiny task gaps.** RTE/WSC are small and noisy; report
    task concentration, seed status, and the predeclared decision rule.

## 8. Settled experimental findings (do not reopen without new evidence)

- AdamW stays; Muon tied zero-shot but lost GLUE through MultiRC collapse.
- 16k vocabulary stays; 8k was refuted across seeds.
- At fixed parameters, depth beats width.
- Dropout benefit is scale-dependent: helpful at larger scale, harmful around
  the old 33M setting.
- GLUE at 10M is data-bound; extra capacity changes little.
- Narrow/deep around 35M was a wash after seeding.
- Seed-sensitive wins driven by Supplement or Entity are provisional until
  replicated.

## 9. Immediate open work

- Rerun both existing 10M hybrid AoA evaluations with `min_context=0`; confirm
  the new raw metadata records the setting and that `aoa.csv` automatically
  selects the rerun.
- Verify the official submission packaging rule for causal versus
  bidirectional evaluation. Do not submit a mixed-mode aggregate by accident.
- Keep 10M fallback overall/human-like blank until its AoA submission artifact
  actually exists; its Reading result is already present.
- Update the advisor/submission slide table from current `all_runs.csv`; the
  100M causal `(Super)GLUE` is 69.10, not the bidirectional 69.99.
- Before final submission, run the full backup importer, validator, and unit
  tests, then archive the exact submitted prediction/HF artifacts.
- No result CSV/code changes should be committed until the user reviews the
  rows and explicitly approves the commit.
