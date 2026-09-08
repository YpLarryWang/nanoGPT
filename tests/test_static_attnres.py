"""Static routing correctness and native/HF checkpoint parity."""
import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model import GPT, GPTConfig, attn_res_mix
from eval.hf_nanogpt.modeling_nanogpt import NanoGPTConfig, NanoGPTForCausalLM

class StaticRoutingTest(unittest.TestCase):
    def config(self, static):
        return GPTConfig(n_layer=4,n_head=2,n_embd=16,block_size=8,vocab_size=32,
                         bias=False,use_rmsnorm=True,use_swiglu=True,use_rope=True,
                         use_attn_gate=True,use_attn_res=True,attn_res_block_size=4,
                         use_static_attn_res=static)

    def test_initial_function_and_backbone(self):
        torch.manual_seed(1337); dynamic=GPT(self.config(False)).eval()
        torch.manual_seed(1337); static=GPT(self.config(True)).eval()
        for k,v in static.state_dict().items():
            if 'attn_res_' not in k:
                self.assertTrue(torch.equal(v,dynamic.state_dict()[k]),k)
        x=torch.randint(0,32,(2,8))
        torch.testing.assert_close(static(x,x)[0],dynamic(x,x)[0],rtol=1e-5,atol=1e-6)

    def test_static_mixture_and_gradient(self):
        q=torch.nn.Parameter(torch.tensor([1.,-1.,3.]))
        values=[torch.randn(2,8,16,requires_grad=True) for _ in range(2)]
        got=attn_res_mix(values,q,None)
        expected=sum(w*v for w,v in zip(q[:2].softmax(0),values))
        torch.testing.assert_close(got,expected)
        got.square().mean().backward()
        self.assertGreater(q.grad[:2].abs().sum().item(),0)
        self.assertEqual(q.grad[2].item(),0)
        self.assertTrue(all(v.grad is not None for v in values))

    def test_hf_save_reload_and_logits(self):
        cfg=self.config(True); model=GPT(cfg).eval()
        hf=NanoGPTForCausalLM(NanoGPTConfig(**dataclasses.asdict(cfg))).eval()
        hf.load_state_dict(model.state_dict(),strict=True)
        x=torch.randint(0,32,(2,8))
        expected=model(x,x)[0]
        torch.testing.assert_close(hf(x).logits,expected,rtol=1e-5,atol=1e-6)
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d)
            restored=NanoGPTForCausalLM.from_pretrained(d).eval()
            self.assertTrue(restored.config.use_static_attn_res)
            torch.testing.assert_close(restored(x).logits,expected,rtol=1e-5,atol=1e-6)

if __name__=='__main__': unittest.main()
