"""
Self-contained HuggingFace wrapper for the nanoGPT architecture used in the BabyLM
ablations: LayerNorm/RMSNorm x MLP/SwiGLU x learned-absolute/RoPE positions.

This single file defines BOTH the config and the models so it can be copied verbatim
into a model repo and loaded with `trust_remote_code=True` by the BabyLM 2026
evaluation pipeline:
  * AutoModelForCausalLM -> NanoGPTForCausalLM   (causal zero-shot)
  * AutoModelForMaskedLM -> NanoGPTForMaskedLM   (bidirectional MNTP zero-shot)
  * AutoModel            -> NanoGPTModel         (GLUE fine-tuning backbone)

Module/parameter names mirror nanoGPT's model.py `GPT` exactly, so a nanoGPT
checkpoint state_dict loads with NO key remapping. Two behaviours differ from
model.py on purpose (both required by the eval harness):
  * forward always returns full-sequence logits (B, T, V), not just the last step.
  * attention honours an optional padding `attention_mask` (needed for GLUE's
    left-padded batches; nanoGPT's own forward assumes dense, causal-only inputs).
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, MaskedLMOutput


class NanoGPTConfig(PretrainedConfig):
    model_type = "nanogpt"
    # let generic HF utilities (and the GLUE classifier's `config.hidden_size`) work
    attribute_map = {
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "max_position_embeddings": "block_size",
    }

    def __init__(
        self,
        vocab_size=16384,
        block_size=512,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.0,
        bias=False,
        use_rmsnorm=False,
        use_swiglu=False,
        swiglu_mult=8 / 3,
        use_rope=False,
        use_attn_gate=False,
        use_attn_res=False,
        attn_res_block_size=2,
        bidirectional=False,
        rope_theta=10000.0,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.use_rmsnorm = use_rmsnorm
        self.use_swiglu = use_swiglu
        self.swiglu_mult = swiglu_mult
        self.use_rope = use_rope
        self.use_attn_gate = use_attn_gate
        self.use_attn_res = use_attn_res
        self.attn_res_block_size = attn_res_block_size
        self.bidirectional = bidirectional
        self.rope_theta = rope_theta
        self.layer_norm_epsilon = layer_norm_epsilon
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# --------------------------------------------------------------------------- norms
class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (mirrors nanoGPT)."""

    def __init__(self, ndim, bias, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, input):
        rms = torch.sqrt(input.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (input / rms) * self.weight


def make_norm(config):
    if config.use_rmsnorm:
        return RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
    return LayerNorm(config.n_embd, bias=config.bias, eps=config.layer_norm_epsilon)


def attn_res_mix(sources, q, norm):
    """Softmax attention over depth; normed keys and raw values, matching nanoGPT."""
    values = torch.stack(sources)
    logits = torch.einsum("c,sbtc->sbt", q, norm(values))
    return torch.einsum("sbt,sbtc->btc", logits.softmax(dim=0), values)


# ---------------------------------------------------------------------------- rope
def build_rope_cache(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)
    cos = torch.cat([angles, angles], dim=-1).cos()
    sin = torch.cat([angles, angles], dim=-1).sin()
    return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim); done in fp32 to match nanoGPT
    seq_len = x.size(2)
    dtype = x.dtype
    x = x.float()
    out = x * cos[:seq_len, :] + rotate_half(x) * sin[:seq_len, :]
    return out.to(dtype)


# ----------------------------------------------------------------------- attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_attn_gate = config.use_attn_gate
        self.bidirectional = config.bidirectional
        if self.use_attn_gate:
            self.attn_gate = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.use_rope = config.use_rope
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            assert head_dim % 2 == 0, "RoPE needs an even head_dim"
            cos, sin = build_rope_cache(config.block_size, head_dim, base=config.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        dropout_p = self.dropout if self.training else 0.0
        if attention_mask is None:
            # Dense causal is numerically identical to nanoGPT training. The
            # bidirectional export uses the same weights with no triangular mask.
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p,
                is_causal=not self.bidirectional,
            )
        else:
            keep = attention_mask.to(torch.bool)[:, None, None, :]  # (B, 1, 1, T)
            if self.bidirectional:
                attn_mask = keep
            else:
                causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
                attn_mask = causal[None, None, :, :] & keep
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        if self.use_attn_gate:
            y = y * torch.sigmoid(self.attn_gate(x)) # elementwise input-dependent gate on attn output (Qwen)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ----------------------------------------------------------------------------- mlp
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class SwiGLU(nn.Module):
    """Gated MLP, SiLU gate (Shazeer 2020). Hidden snapped to a multiple of 64."""

    def __init__(self, config):
        super().__init__()
        hidden_size = int(config.swiglu_mult * config.n_embd)
        hidden_size = round(hidden_size / 64) * 64
        self.c_gate = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
        self.c_val = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
        self.c_proj = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        return self.dropout(self.c_proj(F.silu(self.c_gate(input)) * self.c_val(input)))


def make_mlp(config):
    return SwiGLU(config) if config.use_swiglu else MLP(config)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config)
        self.mlp = make_mlp(config)
        self.use_attn_res = config.use_attn_res
        if config.use_attn_res:
            self.block_start = (2 * layer_idx) % config.attn_res_block_size == 0
            self.attn_res_q1 = nn.Parameter(torch.zeros(config.n_embd))
            self.attn_res_norm1 = RMSNorm(config.n_embd)
            self.attn_res_q2 = nn.Parameter(torch.zeros(config.n_embd))
            self.attn_res_norm2 = RMSNorm(config.n_embd)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_attn_res(self, blocks, partial, attention_mask=None):
        h = attn_res_mix(blocks + [partial], self.attn_res_q1, self.attn_res_norm1)
        if self.block_start:
            blocks = blocks + [partial]
            partial = None
        attn_out = self.attn(self.ln_1(h), attention_mask)
        partial = attn_out if partial is None else partial + attn_out
        h = attn_res_mix(blocks + [partial], self.attn_res_q2, self.attn_res_norm2)
        partial = partial + self.mlp(self.ln_2(h))
        return blocks, partial


# ---------------------------------------------------------------- shared backbone
def _build_transformer(config):
    transformer = nn.ModuleDict(
        dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f=make_norm(config),
        )
    )
    if not config.use_rope:
        transformer.wpe = nn.Embedding(config.block_size, config.n_embd)
    return transformer


def _init_attn_res(model, config):
    if not config.use_attn_res:
        return
    size = config.attn_res_block_size
    assert size >= 2 and size % 2 == 0
    assert (2 * config.n_layer) % size == 0
    model.attn_res_qf = nn.Parameter(torch.zeros(config.n_embd))
    model.attn_res_normf = RMSNorm(config.n_embd)


def _transformer_forward(model, transformer, config, input_ids, attention_mask):
    _, t = input_ids.size()
    tok_emb = transformer.wte(input_ids)
    if config.use_rope:
        x = transformer.drop(tok_emb)
    else:
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        x = transformer.drop(tok_emb + transformer.wpe(pos))
    if config.use_attn_res:
        blocks, partial = [], x
        for block in transformer.h:
            blocks, partial = block.forward_attn_res(blocks, partial, attention_mask)
        x = attn_res_mix(blocks + [partial], model.attn_res_qf, model.attn_res_normf)
    else:
        for block in transformer.h:
            x = block(x, attention_mask)
    x = transformer.ln_f(x)
    return x


class NanoGPTPreTrainedModel(PreTrainedModel):
    config_class = NanoGPTConfig
    base_model_prefix = "transformer"
    main_input_name = "input_ids"
    _no_split_modules = ["Block"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class NanoGPTModel(NanoGPTPreTrainedModel):
    """Backbone returning last_hidden_state (AutoModel target; used by GLUE)."""

    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = _build_transformer(config)
        _init_attn_res(self, config)
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = _transformer_forward(self, self.transformer, self.config, input_ids, attention_mask)
        return BaseModelOutput(last_hidden_state=x)


class NanoGPTForCausalLM(NanoGPTPreTrainedModel, GenerationMixin):
    """LM-head model (AutoModelForCausalLM target; used by zero-shot ranking)."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = _build_transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        _init_attn_res(self, config)
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = _transformer_forward(self, self.transformer, self.config, input_ids, attention_mask)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
                ignore_index=-1,
            )
        return CausalLMOutput(loss=loss, logits=logits)


class NanoGPTForMaskedLM(NanoGPTForCausalLM):
    """Same LM head, with unshifted MLM loss for the bidirectional export."""

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = _transformer_forward(self, self.transformer, self.config, input_ids, attention_mask)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return MaskedLMOutput(loss=loss, logits=logits)
