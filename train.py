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
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

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
save_iters = []    # exact iters to archive a separate, weights-only checkpoint
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# data sampling: 'random' = nanoGPT i.i.d. windows (with replacement); 'shuffle' = deterministic
# shuffled-epoch schedule (each token seen once/epoch, per-epoch offset jitter, DDP-correct,
# resumes exactly from sampler_seed). Default 'random' preserves the original nanoGPT behaviour.
sampler = 'random'
sampler_seed = 1337
seed = 1337 # base RNG seed for weight init + 'random' data sampling; vary for multi-seed studies (actual = seed + ddp_rank)
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
use_muon = False

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
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
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
if save_iters_set:
    assert all(isinstance(i, int) and i >= 0 for i in save_iters_set), "save_iters: non-neg ints"
    assert max(save_iters_set) <= max_iters, "a save_iter exceeds max_iters"
    if master_process:
        print(f"will archive {len(save_iters_set)} checkpoints at{sorted(save_iters_set)}")

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

def get_batch(split, scheduled=False):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    if scheduled:                                    # sampler='shuffle' training fetch (exact epochs)
        global train_draw
        ix = torch.from_numpy(schedule.batch_starts(train_draw))
        train_draw += 1
    else:                                            # nanoGPT i.i.d. windows (with replacement)
        ix = torch.randint(len(data) - block_size, (batch_size,))
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

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                  block_size=block_size, bias=bias, 
                  use_rmsnorm=use_rmsnorm, use_swiglu=use_swiglu, swiglu_mult=swiglu_mult, use_rope=use_rope, use_attn_gate=use_attn_gate,
                  vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'swiglu_mult']:
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
    best_val_loss = checkpoint['best_val_loss']
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
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
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
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.define_metric("tokens")
    wandb.define_metric("*", step_metric="tokens")   # plot all metrics vs tokens

# build the deterministic shuffled-epoch schedule (no-op unless sampler='shuffle')
if sampler == 'shuffle':
    train_len = len(np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r'))
    schedule = ShuffleSchedule(train_len, block_size, batch_size, ddp_world_size,
                               ddp_rank if ddp else 0,
                               (max_iters + 3) * gradient_accumulation_steps, sampler_seed)
    train_draw = iter_num * gradient_accumulation_steps    # exact resume: derive draw from iter_num
    if master_process:
        print(f"sampler='shuffle': {schedule.n_epochs} shuffled epochs x {schedule.n_chunks:,} "
              f"chunks, seed={sampler_seed}, start draw={train_draw}")
elif sampler != 'random':
    raise ValueError(f"unknown sampler {sampler!r} (expected 'random' or 'shuffle')")

# training loop
X, Y = get_batch('train', scheduled=(sampler == 'shuffle')) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# single place to write a checkpoint. weights_only=True drops the optimizer (from the series)
def save_checkpoint(path: str, weights_only: bool = False, extra: dict = None):
    ckpt = {
        'model': raw_model.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'config': config
    }
    if not weights_only:
        ckpt['optimizer'] = [opt.state_dict() for opt in optimizers]
        ckpt['best_val_loss'] = best_val_loss
    if extra:
        ckpt.update(extra)
    print(f"saving checkpoint -> {path}")
    torch.save(ckpt, path)

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    frac = lr / learning_rate
    for opt, bases in zip(optimizers, base_lrs):
        for group, base in zip(opt.param_groups, bases):
            group['lr'] = base * frac

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "tokens": iter_num * tokens_per_iter,   # cumulative tokens seen
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': [opt.state_dict() for opt in optimizers],
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    # archive a weights-only snapshot at the exact shceduled iters (incl. step 0)
    if iter_num in save_iters_set and master_process:
        save_checkpoint(os.path.join(out_dir, f"ckpt_{iter_num:06d}.pt"), weights_only=True)

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
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train', scheduled=(sampler == 'shuffle'))
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

    # termination conditions
    if iter_num > max_iters:
        break

# final metrics + one-line experiment record (project ground-truth log)
if master_process:  # only rank-0 writes, so a multi-GPU (DDP) run logs one line, not N.
    final_metrics = estimate_loss()
    # the in-loop eval only fires at multiples of eval_interval, so when max_iters % eval_interval
    # != 0 the final model is never offered to the checkpointer. Give it a save chance here if it
    # beats every periodic eval (no-op when it doesn't). Guard iter_num > 0 so an eval_only / zero-
    # step pass can't checkpoint a random-init model. Scoped to the best-val regime; when
    # always_save_checkpoint is on, the periodic path already saves the latest model every eval.
    if not always_save_checkpoint and iter_num > 0 and final_metrics['val'] < best_val_loss:
        best_val_loss = final_metrics['val']
        save_checkpoint(os.path.join(out_dir, 'ckpt.pt'))  # final model became the new best
    record = {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'run_name': wandb_run_name,
        'dataset': dataset,
        'sampler': sampler,
        'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd, 'block_size': block_size,
        'params_M': round(raw_model.get_num_params() / 1e6, 2),
        'batch_size': batch_size, 'grad_accum': gradient_accumulation_steps,
        'tokens_per_iter': tokens_per_iter,
        'max_iters': max_iters, 'final_iter': iter_num,
        'total_tokens': iter_num * tokens_per_iter,
        'learning_rate': learning_rate, 'min_lr': min_lr, 'warmup_iters': warmup_iters,
        'train_loss': round(final_metrics['train'].item(), 4),  # losses are tensors — convert to plain floats so json.dumps works.
        'val_loss': round(final_metrics['val'].item(), 4),
        'best_val_loss': round(float(best_val_loss), 4),
        'mfu': round(running_mfu, 4),
        'wandb_id': wandb.run.id if wandb_log else None,
    }
    os.makedirs('results', exist_ok=True)
    with open('results/experiments.jsonl', 'a') as f:        # APPEND by setting mode to 'a', never overwrite
        f.write(json.dumps(record) + '\n')
    print(f"logged run -> results/experiments.jsonl  (val_loss={record['val_loss']})")

if wandb_log and master_process:
    wandb.finish()                                            # mark the run 'finished', flush

if ddp:
    destroy_process_group()
