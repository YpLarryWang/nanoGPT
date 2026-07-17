"""
BabyLM 2026 -- Stage 2: byte-level BPE vocab-size sweep + report.  [GIVEN HARNESS]

The tokenizer itself lives in bpe.py (your from-scratch build). This file just
trains a tokenizer per vocab size and reports the 4 metrics. Two backends:
    --impl hf    (default) the Rust library -- fast; use for the real 9M-word sweep
    --impl ours  YOUR data/babylm/bpe.py   -- verified against HF; slow, so run it
                 on a subset/small vocab to see your own code make a real tokenizer

Pipeline:
    clean official train -> clean/train
    clean official dev   -> clean/val
    python data/babylm/train_bpe.py --clean-dir <data-dir>/clean --out-dir <data-dir>/tokenizer
    # see your own code run on real data, small:
    python data/babylm/train_bpe.py  --impl ours --sources childes --vocab-sizes 8000
Lesson: tutorials/teach/lessons/0002-tokenizer.html
"""
import argparse
import hashlib
import json
import os

try:
    from .constants import EOT, SOURCES
except ImportError:  # direct script execution
    from constants import EOT, SOURCES

VOCAB_SIZES = [8000, 16000, 32000]

# --- baby-GPT config: used ONLY for the embedding-param-share metric ----------
N_EMBD, N_LAYER = 512, 8
# non-embedding params (bias=False, MLP 4x, norm~0): attn 4*d^2 + mlp 8*d^2 per block
NON_EMBD = N_LAYER * 12 * N_EMBD * N_EMBD


# --- tokenizer backends [GIVEN] ----------------------------------------------
def build_and_train(impl, vocab_size, paths, add_prefix_space=True):
    """Train a byte-level BPE on `paths` (train split only). Returns an object with
    .encode(str).ids / .encode_batch(list) -- both bpe.Tokenizer and HF satisfy it."""
    if impl == "ours":
        import bpe
        tok = bpe.Tokenizer(bpe.ByteLevel(add_prefix_space=add_prefix_space))
        tok.train(paths, bpe.BpeTrainer(
            vocab_size=vocab_size, special_tokens=[EOT],
            initial_alphabet=bpe.ByteLevel.alphabet()))
    else:
        from tokenizers import Tokenizer, pre_tokenizers, decoders
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        tok = Tokenizer(BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        tok.decoder = decoders.ByteLevel()
        tok.train(paths, BpeTrainer(
            vocab_size=vocab_size, special_tokens=[EOT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tok


def save_tokenizer(tok, impl, out_dir, vocab_size):
    os.makedirs(out_dir, exist_ok=True)
    if impl == "ours":
        return tok.save(out_dir, prefix=f"bpe-{vocab_size}")          # vocab.json + merges.txt
    path = os.path.join(out_dir, f"bpe-{vocab_size}.json")
    tok.save(path)                                                    # single tokenizer.json
    return path


def train_paths(clean_dir, sources):
    paths = []
    for s in sources:
        p = os.path.join(clean_dir, "train", s + ".txt")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- clean the complete official train split first")
        paths.append(p)
    return paths


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    ap.add_argument("--impl", choices=["hf", "ours"], default="hf")
    ap.add_argument("--sources", nargs="*", default=SOURCES)
    ap.add_argument("--vocab-sizes", nargs="*", type=int, default=VOCAB_SIZES)
    ap.add_argument("--add-prefix-space", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    paths = train_paths(args.clean_dir, args.sources)
    n_words = count_words(paths)
    print(f"impl={args.impl}  train words: {n_words:,}\n")

    rows = []
    outputs = []
    for V in args.vocab_sizes:
        tok = build_and_train(args.impl, V, paths, add_prefix_space=args.add_prefix_space)
        dest = save_tokenizer(tok, args.impl, args.out_dir, V)
        n_tok = count_tokens(tok, paths)
        emb = V * N_EMBD
        rows.append((V, n_tok, n_words / n_tok, avg_probe_tokens(tok),
                     emb, emb / (emb + NON_EMBD)))
        outputs.append({"vocab_size": V, "path": os.path.basename(dest),
                        "sha256": sha256_file(dest)})
        print(f"saved {dest}")

    manifest = {
        "schema_version": 1,
        "protocol": "official-train-dev-v1",
        "tokenizer_inputs": [
            {"path": os.path.abspath(path), "sha256": sha256_file(path)}
            for path in paths
        ],
        "dev_in_tokenizer_inputs": False,
        "train_words": n_words,
        "implementation": args.impl,
        "add_prefix_space": args.add_prefix_space,
        "special_tokens": [EOT],
        "outputs": outputs,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "tokenizer_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\n{'vocab':>7} {'tokens':>14} {'words/tok':>10} "
          f"{'probe avg':>10} {'emb params':>12} {'emb share':>10}")
    for V, n_tok, wpt, probe, emb, share in rows:
        print(f"{V:>7} {n_tok:>14,} {wpt:>10.3f} {probe:>10.2f} {emb:>12,} {share:>9.1%}")


if __name__ == "__main__":
    main()
