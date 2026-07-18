"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import json
import hashlib
import os
import time
import math
import pickle
import platform
import random
import socket
import subprocess
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from masked_data import MaskedData
from checkpoint_schedule import CheckpointSchedule, at_update_budget, rounded_word_tag
from data.babylm.constants import (
    validate_offdev_checkpoint_schedule,
    validate_offdev_wandb_name,
)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
resume_checkpoint = '' # explicit checkpoint path; legacy resume falls back to out_dir/ckpt.pt
resume_strict = True # fail closed if a full checkpoint or trajectory-affecting config is incomplete/mismatched
save_iters = []    # exact iters to archive a separate, weights-only checkpoint
checkpoint_schedule = '' # dual word/token schedule JSON; its iters are unioned with save_iters
experiment_log_path = 'results/experiments.jsonl'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
wandb_run_id = '' # required by strict W&B resume; display names are not unique identifiers
wandb_resume = '' # '', 'allow', 'must', 'never', or 'auto'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
eval_batch_size = -1 # <=0 follows batch_size; formal BabyLM runs fix this independently
block_size = 1024
# data sampling: 'random' = nanoGPT i.i.d. windows (with replacement); 'shuffle' = deterministic
# shuffled-epoch schedule (each token seen once/epoch, per-epoch offset jitter, DDP-correct,
# resumes exactly from sampler_seed). Default 'random' preserves the original nanoGPT behaviour.
sampler = 'random'
sampler_seed = 1337
seed = 1337 # base RNG seed for weight init + 'random' data sampling; vary for multi-seed studies (actual = seed + ddp_rank)
eval_seed = 424242 # fixed validation windows/corruption, isolated from training RNG
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
use_rmsnorm=False
use_swiglu=False
swiglu_mult=8/3
use_rope=False
use_attn_gate=False
use_attn_res = False
attn_res_block_size = 2

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# muon optimizer
use_muon = False

# training objective
use_hybrid = False        # GPT-BERT hybrid objective: mix causal + masked microsteps
causal_microsteps = 1     # causal microsteps per accumulation cycle; the rest run masked (ref = 1 of 16)

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # re-read the values (for wandb), will be useful for logging
config['save_iters'] = list(save_iters) # list-valued config is intentionally not in config_keys
if eval_batch_size <= 0:
    eval_batch_size = batch_size
config['eval_batch_size'] = eval_batch_size
assert eval_batch_size > 0
validate_offdev_wandb_name(dataset, wandb_log, wandb_run_name)
validate_offdev_checkpoint_schedule(dataset, wandb_log, checkpoint_schedule)
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
global_gradient_accumulation_steps = gradient_accumulation_steps
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# validate the checkpoint schedule up front
save_iters_set = set(save_iters)
exposure_schedule = None
if checkpoint_schedule:
    exposure_schedule = CheckpointSchedule.load(
        checkpoint_schedule,
        max_iters=max_iters,
        tokens_per_iter=tokens_per_iter,
    )
    save_iters_set.update(exposure_schedule.save_iters)
    schedule_parameters = exposure_schedule.metadata.get('parameters', {})
    expected_schedule_parameters = {
        'block_size': block_size,
        'batch_size': batch_size,
        'global_grad_accum': global_gradient_accumulation_steps,
        'world_size': ddp_world_size,
        'sampler_seed': sampler_seed,
    }
    for name, expected in expected_schedule_parameters.items():
        actual = schedule_parameters.get(name)
        if actual != expected:
            raise ValueError(
                f"checkpoint schedule {name}={actual!r} does not match run {name}={expected!r}"
            )
    config['checkpoint_schedule'] = checkpoint_schedule
    config['checkpoint_schedule_metadata'] = exposure_schedule.metadata.get('fingerprints', {})
if save_iters_set:
    assert all(isinstance(i, int) and i >= 0 for i in save_iters_set), "save_iters: non-neg ints"
    assert max(save_iters_set) <= max_iters, "a save_iter exceeds max_iters"
    if master_process:
        print(f"will archive {len(save_iters_set)} checkpoints at {sorted(save_iters_set)}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

if exposure_schedule is not None:
    def sha256_file(path, chunk_size=8 * 1024 * 1024):
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                digest.update(chunk)
        return digest.hexdigest()

    fingerprints = exposure_schedule.metadata.get('fingerprints', {})
    for filename, key in (
        ('train.bin', 'train_bin_sha256'),
        ('val.bin', 'val_bin_sha256'),
    ):
        expected = fingerprints.get(key)
        if expected:
            actual = sha256_file(os.path.join(data_dir, filename))
            if actual != expected:
                raise ValueError(
                    f"checkpoint schedule {key}={expected} does not match {data_dir}/{filename}={actual}"
                )

class ShuffleSchedule:
    """Deterministic shuffled-epoch schedule over train.bin chunk starts (sampler='shuffle').

    Each epoch is a full pass: every valid block_size window appears exactly once, in shuffled
    order, with a per-epoch random start offset (boundary jitter so fixed blocks aren't memorised
    across the ~10 passes). The run reads a flat concatenation of epochs indexed by (draw, rank),
    so the schedule is stateless: a fixed seed reproduces it and resume is exact via iter_num (no
    sampler state in the checkpoint). Under DDP every rank builds the identical schedule and reads
    disjoint groups, so the ranks tile the stream with no overlap and no gaps.
    """
    def __init__(self, data_len, block_size, batch_size, world_size, rank, n_draws, seed):
        self.B, self.world, self.rank = batch_size, world_size, rank
        gen = torch.Generator().manual_seed(seed)
        max_start = data_len - block_size - 1                    # y needs one token past x
        need = (n_draws * world_size + world_size) * batch_size  # chunks to cover, + one draw slack
        parts, total, epochs, n = [], 0, 0, 0
        while total < need:
            off = int(torch.randint(block_size, (1,), generator=gen))         # per-epoch phase jitter
            n = (max_start - off) // block_size + 1
            starts = off + block_size * torch.arange(n, dtype=torch.int64)
            parts.append(starts[torch.randperm(n, generator=gen)])            # shuffle this epoch
            total += n; epochs += 1
        self.sched = torch.cat(parts).numpy()
        self.n_epochs, self.n_chunks = epochs, n

    def batch_starts(self, draw):
        g = draw * self.world + self.rank                        # this rank's group at this draw
        i = g * self.B
        assert i + self.B <= len(self.sched), "shuffle schedule exhausted -- increase slack"
        return self.sched[i:i + self.B]

def get_batch(split, scheduled=False, batch_size_override=None, generator=None):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    B = batch_size if batch_size_override is None else batch_size_override
    if scheduled:                                    # sampler='shuffle' training fetch (exact epochs)
        assert B == batch_size, "scheduled training batches use the configured microbatch size"
        global train_draw
        ix = torch.from_numpy(schedule.batch_starts(train_draw))
        train_draw += 1
    else:                                            # nanoGPT i.i.d. windows (with replacement)
        ix = torch.randint(len(data) - block_size, (B,), generator=generator)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
best_iter_num = None
resume_rng_state = None
resume_train_draw = None
resume_batch = None
resume_start_iter = None
resume_checkpoint_role = None
train_draw = 0     # next scheduled train batch on this rank (sampler='shuffle'; derived from iter_num)
schedule = None    # ShuffleSchedule, built just before the training loop when sampler='shuffle'

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# GPT-BERT hybrid objective: validate up front, then build the masked-batch source.
# MASK_ID is the first id past the real tokens; random replacements are drawn from the real vocab.
if use_hybrid:
    assert not ddp, "hybrid v1 is single-GPU: the mix lives in the microsteps, not the ranks"
    assert meta_vocab_size is not None and 'eot_id' in meta, "hybrid needs meta.pkl with vocab_size + eot_id"
    assert 0 <= causal_microsteps <= gradient_accumulation_steps, "causal_microsteps out of range"

    eot_id = meta['eot_id']
    MASK_ID = meta_vocab_size
    REAL_VOCAB = meta_vocab_size

    mdata = MaskedData(
        data_dir=data_dir,
        block_size=block_size,
        batch_size=batch_size,
        device=device,
        device_type=device_type,
        eot_id=eot_id,
        mask_id=MASK_ID,
        real_vocab=REAL_VOCAB,
    )
    hybrid_train_generator = torch.Generator().manual_seed(seed + seed_offset + 1_000_003)
else:
    hybrid_train_generator = None

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                  block_size=block_size, bias=bias, 
                  use_rmsnorm=use_rmsnorm, use_swiglu=use_swiglu, swiglu_mult=swiglu_mult, use_rope=use_rope, use_attn_gate=use_attn_gate,
                  use_attn_res=use_attn_res, attn_res_block_size=attn_res_block_size,
                  vocab_size=None, dropout=dropout) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    # model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    if use_hybrid:
        model_args['vocab_size'] = (meta_vocab_size // 64 + 1) * 64   # 16064: room for MASK_ID=16000
    else:
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = resume_checkpoint or os.path.join(out_dir, 'ckpt.pt')
    # Full training checkpoints are trusted, self-generated artifacts and contain
    # optimizer/RNG objects that PyTorch's weights_only=True loader intentionally rejects.
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    required_resume_keys = {
        'model', 'model_args', 'optimizer', 'iter_num', 'best_val_loss',
        'best_iter_num', 'rng_state', 'train_draw', 'prefetched_batch',
        'config', 'provenance',
    }
    missing_resume_keys = sorted(required_resume_keys - checkpoint.keys())
    if missing_resume_keys:
        raise ValueError(
            f"resume checkpoint is not a complete training checkpoint; missing {missing_resume_keys}. "
            "A weights-only milestone cannot resume training."
        )

    checkpoint_config = checkpoint['config']
    resume_exact_keys = (
        'dataset', 'batch_size', 'gradient_accumulation_steps', 'block_size',
        'sampler', 'sampler_seed', 'seed', 'eval_seed',
        'n_layer', 'n_head', 'n_embd', 'dropout', 'bias',
        'use_rmsnorm', 'use_swiglu', 'swiglu_mult', 'use_rope',
        'use_attn_gate', 'use_attn_res', 'attn_res_block_size',
        'learning_rate', 'max_iters', 'weight_decay', 'beta1', 'beta2',
        'grad_clip', 'decay_lr', 'warmup_iters', 'lr_decay_iters', 'min_lr',
        'use_muon', 'use_hybrid', 'causal_microsteps',
        'dtype', 'compile', 'eval_interval', 'eval_iters', 'eval_batch_size',
        'eval_only', 'always_save_checkpoint', 'checkpoint_schedule', 'save_iters',
        'checkpoint_schedule_metadata',
    )
    resume_mismatches = []
    if resume_strict:
        if ddp:
            resume_mismatches.append(
                "strict resume currently supports world_size=1 only; per-rank RNG/batches are not checkpointed"
            )
        if dtype == 'float16':
            resume_mismatches.append(
                "strict float16 resume is unavailable because legacy checkpoints do not contain GradScaler state"
            )
        if compile and use_hybrid:
            resume_mismatches.append(
                "strict compiled hybrid resume is unavailable because all causal/masked graphs cannot be prewarmed safely"
            )
        for key in resume_exact_keys:
            if key not in checkpoint_config and key not in config:
                continue
            if key not in checkpoint_config:
                resume_mismatches.append(f"{key}: missing from checkpoint config")
            elif checkpoint_config[key] != config.get(key):
                resume_mismatches.append(
                    f"{key}: checkpoint={checkpoint_config[key]!r}, run={config.get(key)!r}"
                )
        checkpoint_provenance = checkpoint['provenance']
        for key, current in (
            ('world_size', ddp_world_size),
            ('global_grad_accum', global_gradient_accumulation_steps),
        ):
            if checkpoint_provenance.get(key) != current:
                resume_mismatches.append(
                    f"provenance.{key}: checkpoint={checkpoint_provenance.get(key)!r}, run={current!r}"
                )
        if resume_mismatches:
            raise ValueError(
                "strict resume configuration mismatch:\n  - " + "\n  - ".join(resume_mismatches)
            )

    if not isinstance(checkpoint['train_draw'], int) or checkpoint['train_draw'] < 0:
        raise ValueError(f"invalid resume train_draw={checkpoint['train_draw']!r}")
    prefetched = checkpoint['prefetched_batch']
    if not isinstance(prefetched, (tuple, list)) or len(prefetched) != 2:
        raise ValueError("resume prefetched_batch must contain exactly (X, Y)")
    expected_batch_shape = (batch_size, block_size)
    if any(tuple(tensor.shape) != expected_batch_shape for tensor in prefetched):
        raise ValueError(
            f"resume prefetched_batch shapes {[tuple(t.shape) for t in prefetched]} "
            f"do not match {expected_batch_shape}"
        )
    rng_state = checkpoint['rng_state']
    required_rng_keys = {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'hybrid_train'}
    if not isinstance(rng_state, dict) or required_rng_keys - rng_state.keys():
        raise ValueError(
            f"resume RNG state is incomplete; missing {sorted(required_rng_keys - set(rng_state or {}))}"
        )
    checkpoint_model_args = checkpoint['model_args']

    if use_hybrid:
        assert checkpoint_model_args['vocab_size'] > MASK_ID, \
            "hybrid resume needs a hybrid-born ckpt (vocab must have room for MASK_ID)"

    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    ARCH_KEYS = [
        'n_layer', 'n_head', 'n_embd', 'block_size',
        'bias', 'vocab_size',
        'use_rmsnorm',
        'use_swiglu', 'swiglu_mult',
        'use_rope',
        'use_attn_gate',
        'use_attn_res', 'attn_res_block_size',
    ]
    for k in ARCH_KEYS:
        if k in checkpoint_model_args: # must have this condition or old ckpt would trigger KeyError
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    if not isinstance(iter_num, int) or not 0 <= iter_num <= max_iters:
        raise ValueError(f"resume iter_num={iter_num!r} is outside [0, {max_iters}]")
    resume_start_iter = iter_num
    resume_checkpoint_role = checkpoint.get('checkpoint_role')
    best_val_loss = checkpoint['best_val_loss']
    best_iter_num = checkpoint.get('best_iter_num', iter_num)
    resume_rng_state = checkpoint.get('rng_state')
    resume_train_draw = checkpoint.get('train_draw')
    resume_batch = checkpoint.get('prefetched_batch')
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
# scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizers = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, use_muon)
# capture base LRs BEFORE any resume-load
base_lrs = [[g['lr'] for g in opt.param_groups] for opt in optimizers]
if init_from == 'resume':
    saved = checkpoint['optimizer']
    saved = saved if isinstance(saved, list) else [saved] # legacy single-opt ckpt
    assert len(saved) == len(optimizers), \
        f"resume mismatch: ckpt has {len(saved)} optimizer(s), model built {len(optimizers)} — did use_muon change?"
    for opt, sd in zip(optimizers, saved):
        opt.load_state_dict(sd)
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# torch.compile is lazy: its first eval/train forwards and first backward can
# initialize compiled RNG machinery. If that happens after checkpoint RNG is
# restored, dropout resumes from a shifted Philox state. Materialize the exact
# formal eval and train graphs first, discard the warmup gradients, and only then
# restore checkpoint RNG below. This changes no parameter or optimizer state.
if init_from == 'resume' and compile:
    print("prewarming compiled eval/train graphs before restoring checkpoint RNG...")
    warm_eval_generator = torch.Generator().manual_seed(eval_seed + 1)
    warm_eval_X, warm_eval_Y = get_batch(
        'val',
        batch_size_override=eval_batch_size,
        generator=warm_eval_generator,
    )
    warm_train_X, warm_train_Y = (tensor.to(device) for tensor in resume_batch)
    model.eval()
    with torch.no_grad(), ctx:
        model(warm_eval_X, warm_eval_Y)
    model.train()
    with ctx:
        if use_hybrid:
            _, warm_loss = model(
                warm_train_X,
                warm_train_Y,
                is_causal=causal_microsteps > 0,
            )
        else:
            _, warm_loss = model(warm_train_X, warm_train_Y)
        warm_loss = warm_loss / gradient_accumulation_steps
    scaler.scale(warm_loss).backward()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del warm_eval_X, warm_eval_Y, warm_train_X, warm_train_Y, warm_loss
    print("compiled resume prewarm complete")

# Restore stochastic state only after model/optimizer construction has consumed its
# initialization randomness, and before the first training batch is fetched.
if resume_rng_state is not None:
    random.setstate(resume_rng_state['python'])
    np.random.set_state(resume_rng_state['numpy'])
    torch.set_rng_state(resume_rng_state['torch_cpu'].cpu())
    if torch.cuda.is_available() and resume_rng_state.get('torch_cuda') is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in resume_rng_state['torch_cuda']])
    if hybrid_train_generator is not None and resume_rng_state.get('hybrid_train') is not None:
        hybrid_train_generator.set_state(resume_rng_state['hybrid_train'].cpu())
    print(f"restored RNG state from {ckpt_path}")

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    """Estimate an arbitrarily accurate loss over either split using many batches.
    
    "Arbitrarily accurate" means: you can make the estimate as precise as you want simply by increasing eval_iters — there's no fixed ceiling on accuracy, you just trade compute for precision.

    Returns:
        dict: mean loss over `eval_iters` batches with keys `train` or `val`.
    """
    out = {}
    model.eval()
    for split_index, split in enumerate(['train', 'val']):
        sample_generator = torch.Generator().manual_seed(eval_seed + split_index)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(
                split,
                batch_size_override=eval_batch_size,
                generator=sample_generator,
            )
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    if use_hybrid:
        # Evaluate the masked objective even for the all-causal hybrid control.
        # This is validation-only and does not mean masked batches entered training.
        losses = torch.zeros(eval_iters)
        sample_generator = torch.Generator().manual_seed(eval_seed + 2)
        corruption_generator = torch.Generator().manual_seed(eval_seed + 3)
        for k in range(eval_iters):
            X, Y = mdata.get_masked_batch(
                'val',
                batch_size_override=eval_batch_size,
                sample_generator=sample_generator,
                corruption_generator=corruption_generator,
            )
            with ctx:
                _, loss = model(X, Y, is_causal=False)
            losses[k] = loss.item()
        out['val_masked'] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    if init_from == 'resume' and resume_strict:
        if not wandb_run_id:
            raise ValueError("strict W&B resume requires --wandb_run_id=<original run id>")
        if wandb_resume != 'must':
            raise ValueError("strict W&B resume requires --wandb_resume=must")
    wandb_kwargs = {
        'project': wandb_project,
        'name': wandb_run_name,
        'config': config,
    }
    if wandb_run_id:
        wandb_kwargs['id'] = wandb_run_id
    if wandb_resume:
        wandb_kwargs['resume'] = wandb_resume
    wandb.init(**wandb_kwargs)
    wandb.define_metric("tokens")
    wandb.define_metric("*", step_metric="tokens")   # plot all metrics vs tokens

# build the deterministic shuffled-epoch schedule (no-op unless sampler='shuffle')
if sampler == 'shuffle':
    train_len = len(np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r'))
    schedule = ShuffleSchedule(train_len, block_size, batch_size, ddp_world_size,
                               ddp_rank if ddp else 0,
                               (max_iters + 3) * gradient_accumulation_steps, sampler_seed)
    train_draw = (
        resume_train_draw
        if resume_train_draw is not None
        else iter_num * gradient_accumulation_steps
    )
    if master_process:
        print(f"sampler='shuffle': {schedule.n_epochs} shuffled epochs x {schedule.n_chunks:,} "
              f"chunks, seed={sampler_seed}, start draw={train_draw}")
elif sampler != 'random':
    raise ValueError(f"unknown sampler {sampler!r} (expected 'random' or 'shuffle')")

# training loop
# Fetch the very first batch. Full checkpoints carry the already-prefetched next
# batch so resume does not skip it or advance any training RNG a second time.
if use_hybrid:
    plan = [
        m < causal_microsteps
        for m in range(gradient_accumulation_steps)
        ]
    print(
        f"hybrid plan: {sum(plan)} causal / {len(plan) - sum(plan)} masked per iter"
    )
    def fetch(causal_step):
        global train_draw
        sched = (sampler == 'shuffle')

        if causal_step:
            return get_batch(
                'train',
                scheduled=(sampler == 'shuffle')
            )
        else: # masked step
            ix = None
            if sched:
                ix = torch.from_numpy(schedule.batch_starts(train_draw))
                train_draw += 1
            return mdata.get_masked_batch(
                'train',
                ix=ix,
                sample_generator=hybrid_train_generator,
                corruption_generator=hybrid_train_generator,
            )
    if resume_batch is None:
        X, Y = fetch(plan[0])
else:
    if resume_batch is None:
        X, Y = get_batch(
            'train',
            scheduled=(sampler == 'shuffle')
        )
if resume_batch is not None:
    X, Y = (tensor.to(device) for tensor in resume_batch)
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

def exposure_at(update: int):
    if exposure_schedule is not None:
        return exposure_schedule.exposure_at(update)
    return {'tokens_seen': update * tokens_per_iter, 'words_seen': None}


def labels_at(update: int):
    if exposure_schedule is None:
        return ()
    return exposure_schedule.labels_at(update)


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'hybrid_train': hybrid_train_generator.get_state() if hybrid_train_generator is not None else None,
    }


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


provenance = {
    'git_sha': command_output(['git', 'rev-parse', 'HEAD']),
    'git_dirty': bool(command_output(['git', 'status', '--porcelain'])),
    'hostname': socket.gethostname(),
    'platform': platform.platform(),
    'torch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    'world_size': ddp_world_size,
    'global_grad_accum': global_gradient_accumulation_steps,
    'local_grad_accum': gradient_accumulation_steps,
    'compile': compile,
    'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
    'tf32_cudnn': torch.backends.cudnn.allow_tf32,
    'flash_sdp_enabled': torch.backends.cuda.flash_sdp_enabled() if torch.cuda.is_available() else None,
    'mem_efficient_sdp_enabled': (
        torch.backends.cuda.mem_efficient_sdp_enabled() if torch.cuda.is_available() else None
    ),
    'math_sdp_enabled': torch.backends.cuda.math_sdp_enabled() if torch.cuda.is_available() else None,
    'data_fingerprints': (
        exposure_schedule.metadata.get('fingerprints', {}) if exposure_schedule is not None else {}
    ),
}

manifest_path = os.path.join(out_dir, 'checkpoint_manifest.json')
if init_from == 'resume' and os.path.exists(manifest_path):
    with open(manifest_path, encoding='utf-8') as f:
        checkpoint_manifest = json.load(f)
else:
    checkpoint_manifest = {
        'schema_version': 1,
        'run_name': wandb_run_name,
        'schedule': checkpoint_schedule or None,
        'provenance': provenance,
        'roles': {},
        'checkpoints': [],
    }


def write_manifest():
    tmp_path = manifest_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_manifest, f, indent=2)
        f.write('\n')
    os.replace(tmp_path, manifest_path)


def record_checkpoint(path, role, weights_only, checkpoint_exposure, checkpoint_labels):
    relative_path = os.path.relpath(path, out_dir)
    entry = {
        'path': relative_path,
        'role': role,
        'iter_num': iter_num,
        **checkpoint_exposure,
        'labels': list(checkpoint_labels),
        'weights_only': weights_only,
    }
    checkpoint_manifest['checkpoints'] = [
        old for old in checkpoint_manifest['checkpoints'] if old.get('path') != relative_path
    ]
    checkpoint_manifest['checkpoints'].append(entry)
    checkpoint_manifest['checkpoints'].sort(key=lambda item: (item['iter_num'], item['path']))
    if role in {'best', 'final', 'latest'}:
        checkpoint_manifest['roles'][role] = relative_path
    write_manifest()


# Single place to write a checkpoint. weights_only=True drops optimizer/RNG state
# from AoA series snapshots. Writes are atomic within the output filesystem.
def save_checkpoint(path: str, role: str, weights_only: bool = False, extra: dict = None):
    checkpoint_exposure = exposure_at(iter_num)
    checkpoint_labels = labels_at(iter_num)
    ckpt = {
        'model': raw_model.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'num_updates': iter_num,
        'checkpoint_role': role,
        'checkpoint_labels': list(checkpoint_labels),
        **checkpoint_exposure,
        'config': config,
        'provenance': provenance,
    }
    if not weights_only:
        ckpt['optimizer'] = [opt.state_dict() for opt in optimizers]
        ckpt['best_val_loss'] = best_val_loss
        ckpt['best_iter_num'] = best_iter_num
        ckpt['rng_state'] = capture_rng_state()
        ckpt['train_draw'] = train_draw
        ckpt['prefetched_batch'] = (X.detach().cpu(), Y.detach().cpu())
    if extra:
        ckpt.update(extra)
    if wandb_log and master_process:
        ckpt['wandb_id'] = wandb.run.id

    # A resumed trajectory may reach a milestone written after its last full
    # checkpoint. Preserve that artifact and prove the replay is bitwise-identical;
    # never silently overwrite evidence from the pre-interruption trajectory.
    if init_from == 'resume' and role == 'milestone' and os.path.exists(path):
        existing = torch.load(path, map_location='cpu', weights_only=False)
        existing_model = existing.get('model')
        current_model = ckpt['model']
        if (
            existing.get('iter_num') != iter_num
            or existing.get('checkpoint_role') != 'milestone'
            or not isinstance(existing_model, dict)
            or existing_model.keys() != current_model.keys()
        ):
            raise RuntimeError(f"existing resume milestone metadata mismatch: {path}")
        mismatched_tensors = []
        for key, current_tensor in current_model.items():
            if not torch.equal(existing_model[key].cpu(), current_tensor.detach().cpu()):
                mismatched_tensors.append(key)
                if len(mismatched_tensors) >= 5:
                    break
        if mismatched_tensors:
            raise RuntimeError(
                f"resume replay diverged at existing milestone {path}; "
                f"first mismatched tensors: {mismatched_tensors}"
            )
        print(f"verified existing milestone bitwise-identical; preserving -> {path}")
        record_checkpoint(path, role, weights_only, checkpoint_exposure, checkpoint_labels)
        return
    print(f"saving checkpoint -> {path}")
    tmp_path = path + '.tmp'
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    record_checkpoint(path, role, weights_only, checkpoint_exposure, checkpoint_labels)


def finalize_best_filename():
    current = checkpoint_manifest['roles'].get('best')
    if not current:
        return
    source = os.path.join(out_dir, current)
    best_entry = next(
        item for item in checkpoint_manifest['checkpoints'] if item['path'] == current
    )
    tag = rounded_word_tag(best_entry.get('words_seen'))
    filename = f"ckpt_best-{tag}-i{best_entry['iter_num']:06d}.pt"
    destination = os.path.join(out_dir, filename)
    if os.path.abspath(source) != os.path.abspath(destination):
        os.replace(source, destination)
        best_entry['path'] = filename
        checkpoint_manifest['roles']['best'] = filename
        write_manifest()

final_metrics = None
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    frac = lr / learning_rate
    for opt, bases in zip(optimizers, base_lrs):
        for group, base in zip(opt.param_groups, bases):
            group['lr'] = base * frac

    at_budget = at_update_budget(iter_num, max_iters)

    # The final in-budget weights are always evaluated, even when max_iters is not
    # divisible by eval_interval. Validation is fixed and therefore comparable.
    if (iter_num % eval_interval == 0 or at_budget) and master_process:
        losses = estimate_loss()
        if at_budget or (iter_num == 0 and eval_only):
            final_metrics = losses
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        resume_boundary_eval = init_from == 'resume' and iter_num == resume_start_iter
        if (
            resume_boundary_eval
            and resume_checkpoint_role == 'best'
            and best_iter_num == iter_num
            and float(losses['val']) != float(best_val_loss)
        ):
            raise RuntimeError(
                f"resume boundary validation changed: checkpoint={float(best_val_loss):.9g}, "
                f"recomputed={float(losses['val']):.9g}"
            )

        # The crashed W&B run already contains the checkpoint-boundary evaluation.
        # Recompute it as a guard, but do not append a duplicate history point.
        if wandb_log and not resume_boundary_eval:
            wandb_metrics = {
                "iter": iter_num,
                "tokens": iter_num * tokens_per_iter,   # cumulative tokens seen
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100, # convert to percentage
            }
            if use_hybrid:
                wandb_metrics["val/masked_loss"] = losses["val_masked"]
            wandb.log(wandb_metrics)
        improved = losses['val'] < best_val_loss
        if improved:
            best_val_loss = losses['val']
            best_iter_num = iter_num
            if iter_num > 0:
                save_checkpoint(os.path.join(out_dir, 'ckpt_best.pt'), role='best')
        elif always_save_checkpoint and iter_num > 0:
            save_checkpoint(os.path.join(out_dir, 'ckpt_latest.pt'), role='latest')
    if iter_num == 0 and eval_only:
        break
    # Archive a single weights file for the union of word- and token-series labels.
    if iter_num in save_iters_set and master_process:
        save_checkpoint(
            os.path.join(out_dir, f"ckpt_{iter_num:06d}.pt"),
            role='milestone',
            weights_only=True,
        )

    # iter_num is completed optimizer updates. W(max_iters) is eligible for best,
    # archived/finalized above, and then training stops before update max_iters+1.
    if at_budget:
        if master_process and iter_num > 0:
            final_exposure = exposure_at(iter_num)
            final_path = os.path.join(
                out_dir,
                f"ckpt_final-{rounded_word_tag(final_exposure['words_seen'])}-i{iter_num:06d}.pt",
            )
            save_checkpoint(
                final_path,
                role='final',
                extra={
                    'final_val_loss': float(final_metrics['val']),
                    'final_train_loss': float(final_metrics['train']),
                },
            )
            finalize_best_filename()
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            if use_hybrid:
                logits, loss = model(X, Y, is_causal=plan[micro_step])
            else:
                logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        if use_hybrid:
            next_step = (micro_step + 1) % gradient_accumulation_steps # the mod is only for the last step, micro_step+1 is out of bound what fetch batch for next formal step
            X, Y = fetch(plan[next_step])
        else:
            X, Y = get_batch(
                'train',
                scheduled=(sampler == 'shuffle'),
            )

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        for opt in optimizers:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    for opt in optimizers: # step each opt
        scaler.step(opt)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

# Final metrics + one-line experiment record (project ground-truth log).
if not eval_only:
    assert iter_num == max_iters, (iter_num, max_iters)
if master_process:  # only rank-0 writes, so a multi-GPU (DDP) run logs one line, not N.
    assert final_metrics is not None
    final_exposure = exposure_at(iter_num)
    record = {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'run_name': wandb_run_name,
        'dataset': dataset,
        'sampler': sampler,
        'optimizer': 'muon+adamw' if use_muon else 'adamw',
        'use_muon': use_muon,
        'use_hybrid': use_hybrid,
        'causal_microsteps': causal_microsteps if use_hybrid else None,
        'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd, 'block_size': block_size,
        'dropout': dropout,
        'use_rmsnorm': use_rmsnorm,
        'use_swiglu': use_swiglu,
        'use_rope': use_rope,
        'use_attn_gate': use_attn_gate,
        'use_attn_res': use_attn_res,
        'params_M': round(raw_model.get_num_params() / 1e6, 2),
        'batch_size': batch_size,
        'grad_accum': global_gradient_accumulation_steps,
        'local_grad_accum': gradient_accumulation_steps,
        'tokens_per_iter': tokens_per_iter,
        'eval_batch_size': eval_batch_size, 'eval_iters': eval_iters,
        'val_tokens_per_eval': eval_batch_size * eval_iters * block_size,
        'seed': seed, 'sampler_seed': sampler_seed,
        'eval_seed': eval_seed,
        'save_iters': sorted(save_iters_set),
        'checkpoint_schedule': checkpoint_schedule or None,
        'max_iters': max_iters, 'final_iter': iter_num,
        'total_tokens': final_exposure['tokens_seen'],
        'total_words': final_exposure['words_seen'],
        'learning_rate': learning_rate, 'min_lr': min_lr, 'warmup_iters': warmup_iters,
        'train_loss': round(final_metrics['train'].item(), 4),  # losses are tensors — convert to plain floats so json.dumps works.
        'val_loss': round(final_metrics['val'].item(), 4),
        'best_val_loss': round(float(best_val_loss), 4),
        'best_iter': best_iter_num,
        'mfu': round(running_mfu, 4),
        'wandb_id': wandb.run.id if wandb_log else None,
        'checkpoint_manifest': manifest_path,
        'provenance': provenance,
    }
    if use_hybrid:
        record['causal_microsteps'] = causal_microsteps
        record['val_masked_loss'] = round(
            final_metrics['val_masked'].item(),
            4,
        )
    experiment_log_dir = os.path.dirname(experiment_log_path)
    if experiment_log_dir:
        os.makedirs(experiment_log_dir, exist_ok=True)
    with open(experiment_log_path, 'a') as f:        # APPEND by setting mode to 'a', never overwrite
        f.write(json.dumps(record) + '\n')
    print(f"logged run -> {experiment_log_path}  (val_loss={record['val_loss']})")

if wandb_log and master_process:
    wandb.finish()                                            # mark the run 'finished', flush

if ddp:
    destroy_process_group()
