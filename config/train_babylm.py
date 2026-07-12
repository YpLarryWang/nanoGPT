# BabyLM baby GPT (n_embd=512, n_layer=8, 16k vocab) -- mirrors config/train_tinystories.py.
# Base config for the 2x2x2 architecture ablation; per-run the arch flags, dataset,
# max_iters, out_dir, and wandb_run_name are set by run_babylm_ablation.sh (or --overrides).
#
#   10M (strict-small):  ~12.2M train tokens -> 10 epochs @ 262,144 tok/iter = 466 iters
#   100M (strict):       set --dataset=babylm_100m --max_iters=<10*tok/262144> --warmup_iters=100

out_dir = 'out-babylm'
eval_interval = 50
eval_iters = 50
log_interval = 10

always_save_checkpoint = False   # keep only the best-val checkpoint per run

wandb_log = True                 # override with --wandb_log=False for smoke tests
wandb_project = 'babylm'
wandb_run_name = 'babylm-run'    # overridden per variant

dataset = 'babylm'               # 'babylm' (10M) or 'babylm_100m'
gradient_accumulation_steps = 8
batch_size = 64
block_size = 512
# tokens_per_iter = 8 * 64 * 512 = 262,144  (same effective batch as the tinystories runs)

# baby GPT
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

# --- architecture ablation flags ---
use_rmsnorm = False
use_swiglu = False
swiglu_mult = 8/3
use_rope = False
use_attn_res = False
attn_res_block_size = 2

# --- optimizer flags ---
# -- AdamW --
learning_rate = 6e-4
max_iters = 466                  # 10M default; 100M overrides to ~5150
lr_decay_iters = 466             # keep == max_iters (Chinchilla)
min_lr = 6e-5
warmup_iters = 40                # ~10% of 466 (scaled down from tinystories' 100)
beta2 = 0.95
# -- Muon --
use_muon = False