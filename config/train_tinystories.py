# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-tinystories'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'tinystories'
wandb_run_name = 'ts-50m-r1-0615'

dataset = 'tinystories'
gradient_accumulation_steps = 8
batch_size = 64
block_size = 512 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0      # pretraining: keep 0 
bias: bool = False
use_rmsnorm: bool = False

learning_rate = 6e-4
max_iters = 6000
lr_decay_iters = 6000 # make equal to max_iters usually
min_lr = 6e-5
beta2 = 0.95

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# save 2^n steps for dev-interp analysis
save_iters = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000]