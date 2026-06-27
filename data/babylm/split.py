"""
BabyLM 2026 -- Stage 1.5: carve the validation split.  [GIVEN -- you don't edit this.]

"Split before you fit anything, including the tokenizer." Runs AFTER clean.py and
BEFORE train_bpe.py, so the BPE only ever sees train. prepare.py (Stage 3) imports
the SAME split_index() here, so the sweep and the final bins agree on the split.

Strategy (Lesson 2 decision -- "carve first, train-only", ~1% val):
  * val = the contiguous TAIL of each source's cleaned file (per-source, so every
    source is represented in val).
  * VAL_FRAC = 0.01.
  * childes / gutenberg carry <|endoftext|> document boundaries, so we snap the cut
    BACKWARD to the previous boundary -- train never ends mid-document, val never
    starts mid-document, and val is guaranteed non-empty. The other four sources
    have no document structure, so they cut at the raw line count.

Usage (on the box where clean/ lives, via the venv python):
    python data/babylm/split.py --clean-dir data/babylm/clean
"""
import argparse
import os

SOURCES = ["bnc_spoken", "childes", "gutenberg",
           "open_subtitles", "simple_wiki", "switchboard"]
EOT = "<|endoftext|>"
VAL_FRAC = 0.01
BOUNDARY_SOURCES = {"childes", "gutenberg"}   # emit EOT -> snap the cut to a boundary


def split_index(lines, source, val_frac=VAL_FRAC):
    """Index i s.t. lines[:i] = train, lines[i:] = val.  The single definition of
    the split, shared by train_bpe.py and prepare.py."""
    n = len(lines)
    cut = max(0, min(int(round(n * (1.0 - val_frac))), n))
    if source in BOUNDARY_SOURCES:
        j = cut
        while j > 0 and lines[j - 1].rstrip("\n") != EOT:
            j -= 1            # walk back to just after the previous boundary
        if j > 0:
            cut = j           # val starts on a whole document (j==0 -> no boundary, keep raw cut)
    return cut


def write_split(clean_dir, source):
    """Carve one source into clean/{train,val}/<source>.txt; return (n_train, n_val)."""
    with open(os.path.join(clean_dir, f"{source}.txt"), encoding="utf-8") as f:
        lines = f.readlines()
    i = split_index(lines, source)
    sizes = {}
    for name, chunk in (("train", lines[:i]), ("val", lines[i:])):
        with open(os.path.join(clean_dir, name, f"{source}.txt"), "w", encoding="utf-8") as f:
            f.writelines(chunk)
        sizes[name] = len(chunk)
    return sizes["train"], sizes["val"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="data/babylm/clean")
    ap.add_argument("--sources", nargs="*", default=SOURCES)
    args = ap.parse_args()
    for name in ("train", "val"):
        os.makedirs(os.path.join(args.clean_dir, name), exist_ok=True)

    print(f"{'source':16s} {'train':>10} {'val':>8}  val%")
    gt = gv = 0
    for source in args.sources:
        nt, nv = write_split(args.clean_dir, source)
        gt += nt
        gv += nv
        tot = nt + nv or 1
        print(f"{source:16s} {nt:>10,} {nv:>8,}  {100 * nv / tot:5.2f}%")
    tot = gt + gv or 1
    print(f"{'TOTAL':16s} {gt:>10,} {gv:>8,}  {100 * gv / tot:5.2f}%")


if __name__ == "__main__":
    main()
