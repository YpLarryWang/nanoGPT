"""CPU contract checks for the AttnRes implementation.

Run from the nanoGPT feature workspace:

    python test_attn_res.py
    python test_attn_res.py --champion-smoke

The default suite is deliberately tiny and requires neither BabyLM data nor a
tokenizer.  ``--champion-smoke`` additionally constructs the previously run
37M d384/L16 gated BabyLM architecture and runs one short forward pass.
"""

import argparse
import importlib.util
import subprocess
import sys
import types

import torch

import model as attnres_model


TOY_SEED = 20260711


def toy_config(use_attn_res):
    """The L4 toy uses every settled champion architectural component."""
    return attnres_model.GPTConfig(
        block_size=16,
        vocab_size=128,
        n_layer=4,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
        use_rmsnorm=True,
        use_swiglu=True,
        use_rope=True,
        use_attn_gate=True,
        use_attn_res=use_attn_res,
        attn_res_block_size=2,
    )


def champion_common_config():
    """Historical bl10m-d384L16-do0.1-gate architecture, without AttnRes."""
    return dict(
        block_size=512,
        vocab_size=16_000,
        n_layer=16,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=False,
        use_rmsnorm=True,
        use_swiglu=True,
        use_rope=True,
        use_attn_gate=True,
    )


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def grad_l1(parameter):
    if parameter.grad is None:
        return 0.0
    return parameter.grad.abs().sum().item()


def load_baseline_model(ref):
    """Load model.py from a git ref without checking out or editing files."""
    try:
        source = subprocess.check_output(
            ["git", "show", f"{ref}:model.py"], text=True
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"cannot load baseline ref {ref!r}; use --baseline-ref <known git ref> "
            "or --skip-control"
        ) from error

    module_name = "_attnres_baseline_model"
    baseline = types.ModuleType(module_name)
    baseline.__file__ = f"git:{ref}:model.py"
    sys.modules[module_name] = baseline
    exec(compile(source, baseline.__file__, "exec"), baseline.__dict__)
    return baseline


def check_control_purity(baseline_ref):
    """Flag-off AttnRes must exactly reproduce a pre-AttnRes model."""
    baseline_module = load_baseline_model(baseline_ref)
    common = dict(
        block_size=16,
        vocab_size=128,
        n_layer=4,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
        use_rmsnorm=True,
        use_swiglu=True,
        use_rope=True,
        use_attn_gate=True,
    )

    torch.manual_seed(TOY_SEED)
    baseline = baseline_module.GPT(baseline_module.GPTConfig(**common)).eval()
    torch.manual_seed(TOY_SEED)
    candidate = attnres_model.GPT(
        attnres_model.GPTConfig(**common, use_attn_res=False, attn_res_block_size=2)
    ).eval()

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    assert baseline_state.keys() == candidate_state.keys()
    for name in baseline_state:
        assert torch.equal(baseline_state[name], candidate_state[name]), name

    torch.manual_seed(TOY_SEED + 1)
    x = torch.randint(0, 128, (2, 7))
    with torch.no_grad():
        baseline_logits, baseline_loss = baseline(x, targets=x)
        candidate_logits, candidate_loss = candidate(x, targets=x)
    assert torch.equal(baseline_logits, candidate_logits)
    assert torch.equal(baseline_loss, candidate_loss)
    print("PASS 1 control purity (state_dict, logits, and loss are bit-identical)")


def check_parameter_delta():
    torch.manual_seed(TOY_SEED)
    off = attnres_model.GPT(toy_config(use_attn_res=False))
    torch.manual_seed(TOY_SEED)
    on = attnres_model.GPT(toy_config(use_attn_res=True))

    expected = (4 * on.config.n_layer + 2) * on.config.n_embd
    actual = parameter_count(on) - parameter_count(off)
    assert actual == expected, (actual, expected)
    queries = [
        parameter
        for name, parameter in on.named_parameters()
        if "attn_res_q" in name
    ]
    assert len(queries) == 2 * on.config.n_layer + 1
    assert all(torch.count_nonzero(query) == 0 for query in queries)
    print(f"PASS 2 parameter delta ({actual} == {expected}); all queries are zero-init")


def check_mix_in_isolation():
    torch.manual_seed(TOY_SEED)
    values = [torch.randn(2, 5, 64) for _ in range(4)]
    norm = attnres_model.RMSNorm(64)

    mixed = attnres_model.attn_res_mix(values, torch.zeros(64), norm)
    assert torch.allclose(mixed, sum(values) / 4, atol=1e-6)

    moved = attnres_model.attn_res_mix(values, torch.randn(64), norm)
    assert not torch.allclose(moved, sum(values) / 4, atol=1e-3)

    values[1] = values[1] * 10
    loud = attnres_model.attn_res_mix(values, torch.zeros(64), norm)
    assert torch.allclose(loud, sum(values) / 4, atol=1e-5)
    print("PASS 3 mix isolation (depth softmax; raw values; zero-q mean)")


def check_bookkeeping():
    torch.manual_seed(TOY_SEED)
    model = attnres_model.GPT(toy_config(use_attn_res=True)).eval()
    x = torch.randint(0, model.config.vocab_size, (2, 7))
    seen = []
    real_mix = attnres_model.attn_res_mix

    def traced_mix(sources, q, norm):
        seen.append(len(sources))
        return real_mix(sources, q, norm)

    attnres_model.attn_res_mix = traced_mix
    try:
        with torch.no_grad():
            model(x, targets=x)
    finally:
        attnres_model.attn_res_mix = real_mix

    expected = [1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert seen == expected, (seen, expected)
    print(f"PASS 4 bookkeeping ({' '.join(map(str, seen))})")


def check_gradient_flow():
    """Verify staged gradients caused by the deliberate zero-query initialization."""
    torch.manual_seed(TOY_SEED)
    model = attnres_model.GPT(toy_config(use_attn_res=True)).train()
    x = torch.randint(0, model.config.vocab_size, (2, 7))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    _, loss = model(x, targets=x)
    loss.backward()
    first = {
        name: grad_l1(parameter)
        for name, parameter in model.named_parameters()
        if "attn_res" in name
    }

    # At q=0, every norm gamma has zero first-step gradient.  The pre-attn
    # pair in layer 0 is permanently dead because its softmax has one source.
    assert first["transformer.h.0.attn_res_q1"] == 0.0
    assert all(value == 0.0 for name, value in first.items() if "norm" in name)
    assert all(
        value > 0.0
        for name, value in first.items()
        if "attn_res_q" in name and name != "transformer.h.0.attn_res_q1"
    ), first

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, targets=x)
    loss.backward()
    second = {
        name: grad_l1(parameter)
        for name, parameter in model.named_parameters()
        if "attn_res" in name
    }
    permanently_dead = {
        "transformer.h.0.attn_res_q1",
        "transformer.h.0.attn_res_norm1.weight",
    }
    zero_after_step = {name for name, value in second.items() if value == 0.0}
    assert zero_after_step == permanently_dead, second
    print("PASS 5 gradient flow (zero-init staging; only layer-0 pre-attn pair stays dead)")


def check_causal_leakage():
    torch.manual_seed(TOY_SEED)
    model = attnres_model.GPT(toy_config(use_attn_res=True)).eval()
    x = torch.randint(0, model.config.vocab_size, (1, model.config.block_size))
    x_changed = x.clone()
    x_changed[0, -1] = (x_changed[0, -1] + 1) % model.config.vocab_size

    with torch.no_grad():
        causal, _ = model(x, targets=x, is_causal=True)
        causal_changed, _ = model(x_changed, targets=x_changed, is_causal=True)
        bidi, _ = model(x, targets=x, is_causal=False)
        bidi_changed, _ = model(x_changed, targets=x_changed, is_causal=False)

    causal_delta = (causal[0, 0] - causal_changed[0, 0]).abs().max().item()
    bidi_delta = (bidi[0, 0] - bidi_changed[0, 0]).abs().max().item()
    assert causal_delta == 0.0, causal_delta
    assert bidi_delta > 0.0, bidi_delta
    print(f"PASS 6 leakage (causal={causal_delta:.1f}; bidi={bidi_delta:.3e})")


def check_muon_routing():
    torch.manual_seed(TOY_SEED)
    model = attnres_model.GPT(toy_config(use_attn_res=True))
    attnres_parameters = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if "attn_res" in name
    }

    muon_and_adam = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        device_type="cpu",
        use_muon=True,
    )
    muon_parameters = {id(parameter) for parameter in muon_and_adam[0].param_groups[0]["params"]}
    assert not (attnres_parameters & muon_parameters)

    adamw = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        device_type="cpu",
        use_muon=False,
    )[0]
    no_decay_parameters = {
        id(parameter)
        for group in adamw.param_groups
        if group["weight_decay"] == 0.0
        for parameter in group["params"]
    }
    assert attnres_parameters <= no_decay_parameters
    print("PASS 7 optimizer routing (AdamW, never Muon or weight decay)")


def check_champion_control_purity(baseline_ref):
    """Repeat control purity on the historical 37M champion-shaped architecture."""
    baseline_module = load_baseline_model(baseline_ref)
    common = champion_common_config()

    torch.manual_seed(TOY_SEED)
    baseline = baseline_module.GPT(baseline_module.GPTConfig(**common)).eval()
    torch.manual_seed(TOY_SEED)
    candidate = attnres_model.GPT(
        attnres_model.GPTConfig(**common, use_attn_res=False, attn_res_block_size=8)
    ).eval()

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    assert baseline_state.keys() == candidate_state.keys()
    for name in baseline_state:
        assert torch.equal(baseline_state[name], candidate_state[name]), name

    torch.manual_seed(TOY_SEED + 1)
    x = torch.randint(0, common["vocab_size"], (1, 8))
    with torch.no_grad():
        baseline_logits, baseline_loss = baseline(x, targets=x)
        candidate_logits, candidate_loss = candidate(x, targets=x)
    assert torch.equal(baseline_logits, candidate_logits)
    assert torch.equal(baseline_loss, candidate_loss)
    print("PASS champion control purity (37M flag-off is bit-identical to master)")


def champion_smoke():
    """One short forward through the historical 37M d384/L16 gate variant."""
    torch.manual_seed(TOY_SEED)
    config = attnres_model.GPTConfig(
        **champion_common_config(),
        use_attn_res=True,
        attn_res_block_size=8,
    )
    model = attnres_model.GPT(config).eval()
    x = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        logits, loss = model(x, targets=x)
    assert logits.shape == (1, 8, config.vocab_size)
    assert torch.isfinite(loss)
    print(
        f"PASS champion smoke ({parameter_count(model) / 1e6:.2f}M params; "
        f"finite loss={loss.item():.4f})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-ref",
        default="master",
        help="pre-AttnRes git ref used by control-purity check (default: master)",
    )
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="skip check 1 when the requested baseline git ref is unavailable",
    )
    parser.add_argument(
        "--champion-smoke",
        action="store_true",
        help="also run one CPU forward through bl10m-d384L16-do0.1-gate (37M)",
    )
    args = parser.parse_args()

    if not args.skip_control:
        check_control_purity(args.baseline_ref)
    check_parameter_delta()
    check_mix_in_isolation()
    check_bookkeeping()
    check_gradient_flow()
    check_causal_leakage()
    check_muon_routing()
    if args.champion_smoke:
        if not args.skip_control:
            check_champion_control_purity(args.baseline_ref)
        champion_smoke()
    print("ALL SELECTED ATTNRES CHECKS PASSED")


if __name__ == "__main__":
    main()
