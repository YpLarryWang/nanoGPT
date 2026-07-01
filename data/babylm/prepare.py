# BabyLM: tokenize the carved clean splits into train.bin/val.bin + meta.pkl.
#   clean/{train,val}/*.txt  --(16k byte-level BPE)-->  uint16 stream + meta.pkl
# Run from repo root (strict-small):  python data/babylm/prepare.py
#   for the 100M strict set:          python data/babylm/prepare.py --data-dir data/babylm_100m

import argparse
import os
import pickle
import numpy as np
from tokenizers import Tokenizer
from split import SOURCES


ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.path.dirname(__file__),
                help="dataset dir holding clean/ + tokenizer/; bins + meta.pkl are written here")
ap.add_argument("--tokenizer", default="bpe-16000.json")
args = ap.parse_args()

DATA_DIR = args.data_dir
TOKENIZER_NAME = args.tokenizer
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer", TOKENIZER_NAME)

EOT_TEXT = "<|endoftext|>"
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
        
#     # # 防御性检查，确认第二遍写入数量和第一遍计数一致
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


if __name__ == '__main__':

    if eot is None:
        raise ValueError(f"{EOT_TEXT} is not registered in tokenizer")

    if tokenizer.get_vocab_size() > np.iinfo(DTYPE).max:
        raise ValueError("vocab size does not fit in uint16")

    for split_name in ("train", "val"):
        write_bin(split_name)

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "eot_id": eot,
        "tokenizer": TOKENIZER_NAME,
    }

    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)