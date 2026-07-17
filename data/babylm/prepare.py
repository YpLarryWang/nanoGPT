# BabyLM: tokenize independent official train/dev clean splits into train.bin/val.bin + meta.pkl.
#   clean/{train,val}/*.txt  --(16k byte-level BPE)-->  uint16 stream + meta.pkl
# Run from repo root (strict-small):  python data/babylm/prepare.py
#   for the 100M strict set:          python data/babylm/prepare.py --data-dir data/babylm_100m

import argparse
import hashlib
import json
import os
import pickle
import numpy as np
from tokenizers import Tokenizer
try:
    from .constants import EOT, SOURCES
except ImportError:  # direct script execution
    from constants import EOT, SOURCES


ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.path.dirname(__file__),
                help="dataset dir holding clean/ + tokenizer/; bins + meta.pkl are written here")
ap.add_argument("--tokenizer", default="bpe-16000.json")
args = ap.parse_args()

DATA_DIR = args.data_dir
TOKENIZER_NAME = args.tokenizer
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer", TOKENIZER_NAME)

EOT_TEXT = EOT
DTYPE = np.uint16


tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
eot = tokenizer.token_to_id(EOT_TEXT)

def iter_token_ids(split_name):
    prev = None
    for src in SOURCES:
        path = f"{DATA_DIR}/clean/{split_name}/{src}.txt"
        with open(path, encoding="utf-8") as f:
            for line in f:
                ids = tokenizer.encode(line.rstrip("\n")).ids
                
                for tok_id in ids:
                    # dedup consecutive EOT tokens
                    if tok_id == eot and prev == eot:
                        continue
                
                    yield tok_id
                    prev = tok_id
                    
        # boundary between sources
        if prev != eot:
            yield eot
            prev = eot


# def count_tokens(split_name):
#     """
#     First pass: count how many tokens this split will contain.

#     memmap needs the final shape before writing.
#     """
#     n = 0
#     for _ in iter_token_ids(split_name):
#         n += 1
#     return n


# def write_bin(split_name, chunk_size=1_000_000):
#     n_tokens = count_tokens(split_name)
#     out_path = f"{DATA_DIR}/{split_name}.bin"
    
#     # create file-backed array
#     arr = np.memmap(
#         out_path,
#         dtype=DTYPE,
#         mode='w+',
#         shape=(n_tokens,),
#     )

#     # small numpy array in RAM
#     buf = np.empty(chunk_size, dtype=np.uint16)
    
#     idx = 0  # position in arr
#     used = 0 # how many space used in buf
    
#     for token_id in iter_token_ids(split_name):
#         # first write in RAM buffer
#         buf[used] = token_id
#         used += 1
        
#         # if buf is full, then write into memmap altogether
#         if used == chunk_size:
#             arr[idx: idx + used] = buf
#             idx += used
#             used = 0 # clear used
            
#     # usually the last chunk is not full so write it into arr as long as it is not empty
#     if used > 0:
#         arr[idx: idx + used] = buf[:used]
#         idx += used
        
#     # # Defensive check to confirm that the number of writes in the second pass is consistent with the count in the first pass.
#     # if idx != n_tokens:
#     #     raise RuntimeError(
#     #         f"{split_name}: wrote {idx} tokens, expected {n_tokens}"
#     #     )
    
#     # flush OS/numpy cache
#     arr.flush()
    
#     print(split_name, n_tokens, "tokens")

def write_bin(split_name):
    arr = np.fromiter(iter_token_ids(split_name), dtype=DTYPE)
    arr.tofile(os.path.join(DATA_DIR, f"{split_name}.bin"))
    print(split_name, arr.size, "tokens")
    return int(arr.size)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == '__main__':

    if eot is None:
        raise ValueError(f"{EOT_TEXT} is not registered in tokenizer")

    if tokenizer.get_vocab_size() > np.iinfo(DTYPE).max:
        raise ValueError("vocab size does not fit in uint16")

    token_counts = {}
    for split_name in ("train", "val"):
        for source in SOURCES:
            path = os.path.join(DATA_DIR, "clean", split_name, f"{source}.txt")
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                raise FileNotFoundError(f"missing or empty clean input: {path}")
        token_counts[split_name] = write_bin(split_name)

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "eot_id": eot,
        "tokenizer": TOKENIZER_NAME,
        "protocol": "official-train-dev-v1",
    }

    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    manifest = {
        "schema_version": 1,
        "protocol": "official-train-dev-v1",
        "tokenizer": {
            "path": os.path.abspath(TOKENIZER_PATH),
            "sha256": sha256_file(TOKENIZER_PATH),
            "vocab_size": tokenizer.get_vocab_size(),
            "eot_id": eot,
        },
        "clean_inputs": {
            split_name: [
                {
                    "source": source,
                    "path": os.path.abspath(os.path.join(DATA_DIR, "clean", split_name, f"{source}.txt")),
                    "sha256": sha256_file(os.path.join(DATA_DIR, "clean", split_name, f"{source}.txt")),
                }
                for source in SOURCES
            ]
            for split_name in ("train", "val")
        },
        "bins": {
            split_name: {
                "path": os.path.abspath(os.path.join(DATA_DIR, f"{split_name}.bin")),
                "tokens": token_counts[split_name],
                "sha256": sha256_file(os.path.join(DATA_DIR, f"{split_name}.bin")),
                "dtype": "uint16",
            }
            for split_name in ("train", "val")
        },
    }
    with open(os.path.join(DATA_DIR, "data_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
