"""
BabyLM 2026 -- Stage 2: custom byte-level BPE + vocab-size sweep.  [LESSON 2 SCAFFOLD]

>>> Fill the ONE TODO block: new_bpe_tokenizer() + new_trainer(). The split, sweep,
>>> metrics, and IO are given. Green the autograder first, then run the sweep:
>>>     python tutorials/test_tokenizer.py
>>>     python data/babylm/split.py      --clean-dir data/babylm/clean      # carve once
>>>     python data/babylm/train_bpe.py  --clean-dir data/babylm/clean      # the sweep
>>> Lesson: tutorials/teach/lessons/0002-tokenizer.html

Why these choices (Lesson 2):
  * byte-level BPE  -> every byte is in the base alphabet, so no <unk>, no OOV ever.
  * NO normalizer   -> clean.py already chose casing per source; a lowercase
                       normalizer here would flatten the case it kept on purpose.
  * ONE special token <|endoftext|> = boundary = eos = pad, registered so the
                       trainer reserves an id for it and it is never split.
  * train on the TRAIN split only (split.py carved it) -> zero val leakage.
"""
import argparse
import os

from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

import split   # the shared carve (same directory)

EOT = split.EOT
VOCAB_SIZES = [8000, 16000, 32000]

# --- baby-GPT config: used ONLY for the embedding-param-share metric ----------
N_EMBD, N_LAYER = 512, 8
# non-embedding params (bias=False, MLP 4x, norm~0): attn 4*d^2 + mlp 8*d^2 per block
NON_EMBD = N_LAYER * 12 * N_EMBD * N_EMBD


# =========================================================================== #
# >>> YOUR TODO -- the conceptual core (~12 lines). Everything below is given. #
# =========================================================================== #
def new_bpe_tokenizer():
    """Return a fresh, UNtrained byte-level BPE Tokenizer.
    Wire up four components (lesson 0002 sections 4-5; HF 'tokenizers' components docs):
      - model        : a BPE model        (byte-level needs no unk_token -- why?)
      - pre_tokenizer: byte-level          (mind add_prefix_space -- section 5 gotcha)
      - decoder      : byte-level          (so decode() inverts the pre-tokenizer)
      - normalizer   : NONE                (do NOT lowercase -- section 4)
    Available: Tokenizer, BPE, pre_tokenizers.ByteLevel, decoders.ByteLevel.
    """
    raise NotImplementedError("implement new_bpe_tokenizer()  (data/babylm/train_bpe.py)")


def new_trainer(vocab_size):
    """Return a BpeTrainer targeting `vocab_size` that:
      - registers EOT as a special token (reserved id + never split), and
      - seeds all 256 bytes as the base alphabet (look at ByteLevel.alphabet()),
        so every byte exists before any merge -> truly no OOV on the full corpus.
    Available: BpeTrainer, pre_tokenizers.ByteLevel.
    """
    raise NotImplementedError("implement new_trainer()  (data/babylm/train_bpe.py)")
# =========================================================================== #


def train_paths(clean_dir, sources=split.SOURCES):
    """The train-split files the BPE learns from (split.py must have run)."""
    paths = []
    for s in sources:
        p = os.path.join(clean_dir, "train", s + ".txt")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run "
                             f"`python data/babylm/split.py --clean-dir {clean_dir}` first")
        paths.append(p)
    return paths


# --- metrics  [GIVEN] ---------------------------------------------------------
EVAL_PROBES = [
    "The keys to the cabinet are on the table.",
    "She did not want to go to the party tonight.",
    "The scientist who wrote the paper is very famous.",
    "I wonder what he bought at the store yesterday.",
    "The children were playing happily in the garden.",
]


def _batched(it, size=10000):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_words(paths):
    n = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                if s and s != EOT:
                    n += len(s.split())
    return n


def count_tokens(tok, paths):
    n = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for chunk in _batched(s.rstrip("\n") for s in f if s.strip()):
                for enc in tok.encode_batch(chunk):
                    n += len(enc.ids)
    return n


def avg_probe_tokens(tok):
    return sum(len(tok.encode(s).ids) for s in EVAL_PROBES) / len(EVAL_PROBES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="data/babylm/clean")
    ap.add_argument("--out-dir", default="data/babylm/tokenizer")
    ap.add_argument("--vocab-sizes", nargs="*", type=int, default=VOCAB_SIZES)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    paths = train_paths(args.clean_dir)
    n_words = count_words(paths)
    print(f"train words: {n_words:,}\n")

    rows = []
    for V in args.vocab_sizes:
        tok = new_bpe_tokenizer()
        tok.train(paths, new_trainer(V))
        dest = os.path.join(args.out_dir, f"bpe-{V}.json")
        tok.save(dest)
        n_tok = count_tokens(tok, paths)
        emb = V * N_EMBD
        rows.append((V, n_tok, n_words / n_tok, avg_probe_tokens(tok),
                     emb, emb / (emb + NON_EMBD)))
        print(f"saved {dest}")

    print(f"\n{'vocab':>7} {'tokens':>14} {'words/tok':>10} "
          f"{'probe avg':>10} {'emb params':>12} {'emb share':>10}")
    for V, n_tok, wpt, probe, emb, share in rows:
        print(f"{V:>7} {n_tok:>14,} {wpt:>10.3f} {probe:>10.2f} {emb:>12,} {share:>9.1%}")


if __name__ == "__main__":
    main()
