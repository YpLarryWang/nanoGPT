#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# (a) One-shot setup for a FRESH Vast.ai PyTorch instance to run this repo.
#
# On a fresh box, either:
#   git clone https://github.com/YpLarryWang/nanoGPT.git /workspace/nanoGPT
#   bash /workspace/nanoGPT/setup_vast.sh
# or one line (public repo):
#   curl -fsSL https://raw.githubusercontent.com/YpLarryWang/nanoGPT/master/setup_vast.sh | bash
#
# Idempotent: safe to re-run. Skips data prep if the .bin files already exist.
# Env knobs:  SKIP_DATA=1  -> don't run prepare.py (do it yourself later)
#
# SECRETS ARE NOT STORED HERE. Put WANDB_API_KEY / HF_TOKEN in ~/.bashrc yourself;
# this script only checks and reminds you.
# ---------------------------------------------------------------------------
set -u
REPO_URL="https://github.com/YpLarryWang/nanoGPT.git"
REPO_DIR="/workspace/nanoGPT"
VENV="/venv/main"

echo "=== [1/5] repo ==="
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR" || { echo "cannot cd to $REPO_DIR"; exit 1; }
git pull --ff-only 2>/dev/null || echo "  (skipped pull — local changes or non-ff)"

echo "=== [2/5] python deps into $VENV ==="
# shellcheck disable=SC1090
source "$VENV/bin/activate"
uv pip install -q tiktoken transformers datasets wandb && echo "  deps OK"

echo "=== [3/5] conda: disable base auto-activate (so /venv/main stays on PATH) ==="
[ -f /opt/miniforge3/etc/profile.d/conda.sh ] && source /opt/miniforge3/etc/profile.d/conda.sh
conda config --set auto_activate_base false 2>/dev/null && echo "  auto_activate_base=false"

echo "=== [4/5] keys check (NOT stored by this script) ==="
[ -n "${WANDB_API_KEY:-}" ] && echo "  WANDB_API_KEY: set" || echo "  WANDB_API_KEY: MISSING (needed for wandb_log=True)"
[ -n "${HF_TOKEN:-}" ]      && echo "  HF_TOKEN: set"      || echo "  HF_TOKEN: missing (optional; only removes HF rate-limit warning)"
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "  -> add to ~/.bashrc, then open a FRESH shell (do NOT 'source ~/.bashrc'):"
  echo "       echo 'export WANDB_API_KEY=\"...\"' >> ~/.bashrc"
  echo "       echo 'export HF_TOKEN=\"...\"'      >> ~/.bashrc"
fi

echo "=== [5/5] data: TinyStories train.bin / val.bin ==="
if [ -f data/tinystories/train.bin ] && [ -f data/tinystories/val.bin ]; then
  echo "  bins already present — skipping prepare.py"
elif [ "${SKIP_DATA:-0}" = 1 ]; then
  echo "  SKIP_DATA=1 — run later: python data/tinystories/prepare.py"
else
  echo "  preparing (downloads + GPT-2 tokenizes, ~10-15 min)..."
  python -u data/tinystories/prepare.py
fi

echo
echo "=== setup done ==="
echo "verify:  python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
echo "then open a FRESH shell so the venv + keys load, and launch training or ./run_sweep_multigpu.sh"
