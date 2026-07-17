# BabyLM official-development data protocol

New runs use the full cleaned official training split and the full cleaned
official development split. The tokenizer is trained on cleaned train only.
The legacy `split.py` tail carve is not part of this workflow.

The two self-contained runtime directories are:

```text
data/babylm_officialdev
data/babylm_100m_officialdev
```

Run on JetStream from the repository root with the training environment:

```bash
PY=/media/volume/yupei-data/envs/nanogpt/bin/python

$PY data/babylm/fetch_offdev.py \
  --track 10m --data-dir data/babylm_officialdev
$PY data/babylm/fetch_offdev.py \
  --track 100m --data-dir data/babylm_100m_officialdev

for data_dir in data/babylm_officialdev data/babylm_100m_officialdev; do
  $PY data/babylm/clean.py --input-split train \
    --raw-dir "$data_dir/raw/train" --out-dir "$data_dir/clean/train"
  $PY data/babylm/clean.py --input-split dev \
    --raw-dir "$data_dir/raw/dev" --out-dir "$data_dir/clean/val"
  $PY data/babylm/audit_offdev.py --data-dir "$data_dir"
  $PY data/babylm/train_bpe.py --clean-dir "$data_dir/clean" \
    --out-dir "$data_dir/tokenizer" --vocab-sizes 16000
  $PY data/babylm/prepare.py --data-dir "$data_dir"
  $PY data/babylm/audit_offdev.py --data-dir "$data_dir"
done
```

The second audit verifies the tokenizer manifest contains exactly the six
`clean/train` inputs, reports train/dev line overlap, and derives `max_iters`
from the measured `train.bin` length. Use that derived value when building new
AoA schedules; never copy 466 or 4740 from the legacy datasets.

All formal run names on these datasets include `-offdev` and use dataset names
`babylm_officialdev` or `babylm_100m_officialdev`.

## Formal AoA schedules

Formal B32/GA16 single-GPU runs always pass a dual actual-word/BPE-token
checkpoint schedule. The committed schedules cover sampler seeds 1337, 1338,
and 1339 for both tracks. Because shuffled word exposure is seed-dependent,
never use one seed's schedule for another seed; `train.py` validates this and
the remaining schedule parameters before training starts.

Use the offdev-only runner for the current `d512L32-do0.1-gate` architecture:

```bash
PY=/media/volume/yupei-data/envs/nanogpt/bin/python \
  CUDA_VISIBLE_DEVICES=0 bash run_babylm_offdev_aoa.sh 10m 1337
PY=/media/volume/yupei-data/envs/nanogpt/bin/python \
  CUDA_VISIBLE_DEVICES=0 bash run_babylm_offdev_aoa.sh 100m 1337
```

The 10M schedule has 19 word labels, 20 token labels, and 37 unique saved
updates. The 100M schedule has 28 word labels, 31 token labels, and 57 unique
saved updates. The final checkpoints are always retained even when the final
exposure is not an integer milestone.
