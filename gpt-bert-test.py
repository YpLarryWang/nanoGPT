# leakage_test.py — run from ~/Documents/experiment/nanoGPT
import torch
from model import GPT, GPTConfig
from masked_data import MaskedData
from tokenizers import Tokenizer


EOT_TEXT = "<|endoftext|>"
tokenizer = Tokenizer.from_file("data/babylm/tokenizer/bpe-16000.json")

def dec(ids):
    return tokenizer.decode(ids)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# tiny model, but exercising YOUR code paths. If any keyword errors, just delete that
# line — the leakage test only depends on is_causal, so the arch flags don't change the verdict.
cfg = GPTConfig(
    block_size=16, vocab_size=128, n_layer=2, n_head=2, n_embd=64,
    dropout=0.0, bias=False,
    use_rmsnorm=True, use_swiglu=True, use_rope=True, use_attn_gate=True,   # ← match your field names
)
m = GPT(cfg).to(device).eval()

x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size), device=device)

with torch.no_grad():
    # pass targets=x so forward returns logits for ALL positions (not just thelast one)
    lc,  _ = m(x,  targets=x,  is_causal=True)      # (1, T, vocab)
    lb,  _ = m(x,  targets=x,  is_causal=False)
    x2 = x.clone(); x2[0, -1] = (x[0, -1] + 1) % cfg.vocab_size   # perturb the LAST token
    lc2, _ = m(x2, targets=x2, is_causal=True)
    lb2, _ = m(x2, targets=x2, is_causal=False)

print("causal  Δ@pos0:", (lc[0, 0] - lc2[0, 0]).abs().max().item())   # expect 0.0
print("bidi    Δ@pos0:", (lb[0, 0] - lb2[0, 0]).abs().max().item())   # expect > 0


md = MaskedData(data_dir='data/babylm', block_size=128, batch_size=16,
                device=device, device_type=('cuda' if 'cuda' in device else 'cpu'),
                eot_id=tokenizer.token_to_id(EOT_TEXT), mask_id=16000, real_vocab=16000)
x, y = md.get_masked_batch('train')
print("md.mask_id in x?  ", (x == md.mask_id).float().mean().item())   # ~0.12  (0.15 × 0.80)
print("md.mask_id in y?  ", (y == md.mask_id).any().item())            # False  (never a target)
print("supervised frac", (y != -100).float().mean().item())      # ~0.15
# where y is supervised, it must equal the ORIGINAL next token (decode a row to eyeball it):
row = 0; sup = (y[row] != -100).nonzero().flatten()[:5]
print("targets:", [dec([y[row,j].item()]) for j in sup])         # real words, not <mask>
# and Mod A + Mod B together: a masked batch through the BIDIRECTIONAL model runs + is finite
big = GPT(GPTConfig(block_size=md.block_size, vocab_size=16064,
                    n_layer=2, n_head=2, n_embd=64, dropout=0.0, bias=False,
                    use_rmsnorm=True, use_swiglu=True, use_rope=True, use_attn_gate=True)
          ).to(device).eval()
_, loss = big(x, targets=y, is_causal=False)
print("masked loss:", loss.item())               # finite, ~9.68 = ln(16064) at init
