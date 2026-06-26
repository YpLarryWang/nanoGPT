"""
BabyLM 2026 -- Stage 1: per-source text cleaning  [LESSON 1 SCAFFOLD].

>>> This is a fill-in-the-blanks scaffold. Find each "TODO" and implement
>>> the per-source cleaners + two regexes. Run the autograder until it's green:
>>>     python tutorials/test_clean_on_samples.py
>>> Lesson: tutorials/teach/lessons/0001-babylm-data-cleaning.html

Cleaning policy (decided together):
    strip speaker prefixes        (*CHI:\\t, *MOT:\\t, A:\\t, ...)
    drop pure annotation lines    ([leaves room.], %mor: tiers)
    normalize unicode / quotes / dashes / whitespace
    unwrap wiki headers           (= = = Title = = =  ->  Title)
    lowercase OpenSubtitles only  (its ALL-CAPS is format noise, not real case)
    keep newlines, no per-row EOT

Architecture: one pure function per source, returning the cleaned line or None to
drop it; a shared normalize() runs on every survivor. Output is per-source so
Lesson 2 can carve a contiguous validation slice from each source.

Usage (run where the raw data lives, e.g. the GPU box):
    python clean.py --raw-dir <snapshot_dir> --out-dir clean
"""

import argparse
import os
import re
import unicodedata
from collections import Counter

SOURCES = ["bnc_spoken", "childes", "gutenberg",
           "open_subtitles", "simple_wiki", "switchboard"]

# --------------------------------------------------------------------------- #
# shared normalization  [GIVEN -- read it; you don't need to change it]
# NFKC folds ligatures / full-width / NBSP, but NOT curly quotes or dashes,
# so we fold those explicitly with _PUNCT_MAP.
# --------------------------------------------------------------------------- #
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    "…": "...",
}
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCT_MAP)))
_WS_RE = re.compile(r"[ \t ]+")

def normalize(s):
    s = unicodedata.normalize("NFKC", s)
    s = _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group()], s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def has_letters_and_digits_and_digits(line):
    return any(c.isalnum() for c in line)

# --------------------------------------------------------------------------- #
# regexes -- THREE are given, TWO are yours to write (see Lesson 1)
# --------------------------------------------------------------------------- #
_CHILDES_TIER = re.compile(r"^%[a-z]{2,4}:")            # %mor:  %gra:        [given]
_BRACKET_ONLY = re.compile(r"^\[[^\]]*\]$")             # [leaves room.]      [given] -- this means to match any number of char that is NOT right bracket
_WIKI_HEADER  = re.compile(r"^(?:=\s)+(.*?)(?:\s=)+$")  # = = = Title = = =   [given]

# v1 (worked, but the full corpus leaked 5,774 line-initial '*'):
#     _CHILDES_SPK = re.compile(r"^\*[A-Za-z]{2,5}:[ \t]+")
# TODO(v2): fix the two leak causes the residual report exposed --
#   (a) numbered codes  "*SI1:" "*SI2:"  -> [A-Za-z] misses the digit  (allow 0-9)
#   (b) bare empty codes "*MOT:"         -> [ \t]+ needs a separator   (make it optional)
_CHILDES_SPK = re.compile(r"^\*[A-Za-z0-9]{2,5}:[ \t]*")

# TODO: a Switchboard speaker prefix at line start -- one capital, ':', a TAB.
#   should match & strip:  "A:\t"  "B:\t"
_SWB_SPK = re.compile(r"^[AB]:\t")

# v2 [given]: an inline annotation span to remove from WITHIN a line --
#   childes "when he [she] naps" / subtitles "hi [bell rings]"  ->  the [..] (+ a leading space)
_INLINE_BRACKET = re.compile(r"\s*\[[^\]]*\]")

# v2: italics and bold format
# _ITALICS = re.compile(r"_[^_]*_")
# _BOLD = re.compile(r"=[^=]*=")
_ITALICS = re.compile(r"_([^_]*)_")
_BOLD = re.compile(r"=([^=]*)=") # add capture group for both format to avoid loop in later replacement

_STAR_BREAK = re.compile(r"^\s*(?:\*\s*){2,}$")
_STAR_WRAPPED_LINE = re.compile(r"^\s*\*([^*\n]*[A-Za-z][^*\n]*)\*\s*$")

# --------------------------------------------------------------------------- #
# per-source cleaners -- return cleaned line, or None to DROP it.
# (these run BEFORE normalize(): strip structure first, then squeeze whitespace)
# --------------------------------------------------------------------------- #

def clean_childes(line):
    # TODO -- three moves, IN THIS ORDER:
    #   1. if line is a %mor:/%gra: tier (_CHILDES_TIER)  -> return None
    #   2. strip a leading "*SPK:\t" prefix               -> _CHILDES_SPK.sub("", line)
    #   3. if what remains is ONLY a [bracketed] note     -> return None  (_BRACKET_ONLY on line.strip())
    #   4. otherwise return line
    line = _CHILDES_SPK.sub("", line)
    if _WIKI_HEADER.match(line.strip()): return None
    if _CHILDES_TIER.match(line.strip()): return None # good practice is to drop annotation first
    if _BRACKET_ONLY.match(line.strip()): return None
    # TODO(v2): strip remaining inline CHAT codes ([//], [: x], [she]) with _INLINE_BRACKET
    line = _INLINE_BRACKET.sub("", line)
    # if not has_letters_and_digits(line): return None
    return line

def clean_switchboard(line):
    # TODO: strip "A:\t"/"B:\t" (_SWB_SPK), drop bracket-only lines, else return line
    line = _SWB_SPK.sub("", line)
    if _BRACKET_ONLY.match(line.strip()): return None
    return line

def clean_subtitles(line):
    line = line.lower()   # ALL-CAPS is format, not meaningful case
    # TODO(v2): remove song markers '♪' and inline caption spans ([bell rings])
    #   -- use _INLINE_BRACKET for captions; a caption-only line then drops to ""
    line = line.replace('♪', '')
    line = _INLINE_BRACKET.sub("", line)
    if not has_letters_and_digits(line): return None
    return line

def clean_wiki(line):
    # TODO: unwrap "= = = Title = = =" -> "Title" with _WIKI_HEADER; else return line
    # if _WIKI_HEADER.match(line.strip()):
    #     return _WIKI_HEADER.match(line.strip()).group(1)
    if (m := _WIKI_HEADER.match(line.strip())): return m.group(1) # good practice is not to run the same code twices
    return line

def clean_gutenberg(line):
    # TODO(v2): Project Gutenberg markup (residual: underscore=2563, equals=166, bracket=1189) --
    #   1. drop "= = = PGxxxxx = = =" book-id headers   (reuse _WIKI_HEADER -> None)  <-- FIRST
    #   2. strip _italics_ and =bold= emphasis markers  (remove '_' and '=' characters)
    if _WIKI_HEADER.match(line.strip()): return None
    line = _INLINE_BRACKET.sub("", line)
    line = _ITALICS.sub(r"\1", line)
    line = _BOLD.sub(r"\1", line)
    if _STAR_BREAK.match(line): return None
    line = _STAR_WRAPPED_LINE.sub(r"\1", line)
    return line

def clean_passthrough(line):
    return line   # bnc_spoken: normalize() alone is enough  [given]

CLEANERS = {
    "bnc_spoken": clean_passthrough,
    "childes": clean_childes,
    "gutenberg": clean_gutenberg,
    "open_subtitles": clean_subtitles,
    "simple_wiki": clean_wiki,
    "switchboard": clean_switchboard,
}

def clean_line(source, raw):
    """A raw line (newline already stripped) -> cleaned line, or None to drop."""
    line = CLEANERS[source](raw)
    if line is None:
        return None
    line = normalize(line)
    if not has_letters_and_digits(line):
        return None
    return line or None

# --------------------------------------------------------------------------- #
# driver + residual-artifact report  [GIVEN -- always inspect your own output]
# --------------------------------------------------------------------------- #
_SUSPECT = {"tab": "\t", "bracket": "[", "star": "*",
            "equals": "=", "angle": "<", "note": "♪", 
            "underscore": "_", "percent": "%"}

def clean_one(raw_path, out_path, source):
    kept = dropped = allcaps = 0
    suspects = Counter()
    with open(raw_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for raw in fin:
            line = clean_line(source, raw.rstrip("\n"))
            if line is None:
                dropped += 1
                continue
            fout.write(line + "\n")
            kept += 1
            for name, ch in _SUSPECT.items():
                if ch in line:
                    suspects[name] += 1
            if line.isupper():
                allcaps += 1
    return kept, dropped, suspects, allcaps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="dir with <source>.train.txt files")
    ap.add_argument("--out-dir", default="clean", help="dir to write <source>.txt files")
    ap.add_argument("--sources", nargs="*", default=SOURCES)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    grand_kept = grand_dropped = 0
    for source in args.sources:
        raw_path = os.path.join(args.raw_dir, f"{source}.train.txt")
        out_path = os.path.join(args.out_dir, f"{source}.txt")
        kept, dropped, suspects, allcaps = clean_one(raw_path, out_path, source)
        grand_kept += kept
        grand_dropped += dropped
        flags = ", ".join(f"{k}={v}" for k, v in suspects.items()) or "none"
        print(f"{source:16s} kept={kept:>7} dropped={dropped:>6} "
              f"| residual: {flags}; allcaps_lines={allcaps}")
    print(f"{'TOTAL':16s} kept={grand_kept:>7} dropped={grand_dropped:>6}")

if __name__ == "__main__":
    main()
