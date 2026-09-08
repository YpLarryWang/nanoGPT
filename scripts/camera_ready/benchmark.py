#!/usr/bin/env python3
"""Synchronized complete-update throughput, without data I/O or evaluation."""
import argparse, dataclasses, json, os, platform, statistics, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from model import GPT,GPTConfig

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','static','dynamic'],required=True)
 p.add_argument('--warmup',type=int,default=10);p.add_argument('--updates',type=int,default=20)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--profile',action='store_true')
 a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(1337);torch.cuda.manual_seed(1337)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 cfg=GPTConfig(block_size=512,vocab_size=16000,n_layer=32,n_head=8,n_embd=512,
 dropout=.1,bias=False,use_rmsnorm=True,use_swiglu=True,use_rope=True,use_attn_gate=True,
 use_attn_res=a.arm!='baseline',use_static_attn_res=a.arm=='static',attn_res_block_size=8)
 raw=GPT(cfg).cuda().train();opt=raw.configure_optimizers(.1,6e-4,(.9,.95),'cuda')
 model=torch.compile(raw)
 x=torch.randint(0,16000,(8,512),device='cuda');y=torch.randint(0,16000,(8,512),device='cuda')
 def update():
  opt.zero_grad(set_to_none=True)
  for _ in range(64):
   with torch.autocast('cuda',dtype=torch.bfloat16): _,loss=model(x,y); loss=loss/64
   loss.backward()
  torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0);opt.step()
  return loss.detach()*64
 for i in range(a.warmup):
  loss=update();torch.cuda.synchronize();print('warmup',i,float(loss),flush=True)
 torch.cuda.reset_peak_memory_stats();times=[]
 for i in range(a.updates):
  torch.cuda.synchronize();t=time.perf_counter();loss=update();torch.cuda.synchronize()
  dt=time.perf_counter()-t;times.append(dt);print('measured',i,dt,float(loss),flush=True)
 q=statistics.quantiles(times,n=4) if len(times)>1 else [times[0]]*3
 d={'arm':a.arm,'seconds':times,'mean_seconds':statistics.mean(times),'median_seconds':statistics.median(times),
 'iqr_seconds':q[2]-q[0],'tokens_per_update':262144,'tokens_per_second':262144/statistics.mean(times),
 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
 'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),
 'gpu':torch.cuda.get_device_name(0),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
 'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'config':dataclasses.asdict(cfg),'warmup':a.warmup,'compile':True,'measurement':'resident-input complete training update; excludes compilation, eval, disk and network',
 'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,power.limit','--format=csv'],text=True)}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(d,f,indent=2)
 if a.profile:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
   update();torch.cuda.synchronize()
  prof.export_chrome_trace(str(a.output.with_suffix('.trace.json')))
  a.output.with_suffix('.profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
if __name__=='__main__':main()
