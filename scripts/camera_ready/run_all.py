#!/usr/bin/env python3
"""Detached two-GPU camera-ready queue. Fail closed; preserve logs/artifacts."""
import concurrent.futures, json, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
STATE=ROOT/'results/camera-ready/queue-state.json'
STATE.parent.mkdir(parents=True,exist_ok=True)
def state(phase,**extra):
 d={'phase':phase,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra}
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(STATE);print(d,flush=True)
def name(seed): return f'bl10m-d512L32-do0.1-gate-attnres8-static-offdev-endpoint-b8ga64-s{seed}'
def call(args,log=None,env=None):
 if log:
  with open(log,'a') as f:subprocess.run(args,check=True,stdout=f,stderr=subprocess.STDOUT,env=env)
 else:subprocess.run(args,check=True,env=env)
def train_lane(seeds,gpu):
 for seed in seeds:call(['bash','scripts/camera_ready/train_seed.sh',str(seed),str(gpu)])
def evaluate(seeds,gpu,phase):
 for seed in seeds:
  call(['bash','scripts/camera_ready/eval_seed.sh',name(seed),str(gpu),phase],f'logs/camera-ready/{name(seed)}.{phase}.log')
def paired(fn,a,b):
 with concurrent.futures.ThreadPoolExecutor(2) as pool:
  futures=[pool.submit(fn,*a),pool.submit(fn,*b)]
  for f in futures:f.result()
def main():
 if STATE.exists():raise RuntimeError('queue already has state; inspect before restarting')
 state('training',gpu0=[1337,1339],gpu1=[1338])
 paired(train_lane,([1337,1339],0),([1338],1))
 state('priority_evaluation')
 paired(evaluate,([1337,1339],0,'priority'),([1338],1,'priority'))
 state('benchmark')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4')
 for repeat,order in enumerate([['baseline','static','dynamic'],['static','dynamic','baseline'],['dynamic','baseline','static']]):
  for arm in order:
   existing=Path(f'results/camera-ready/benchmark-{arm}-{repeat}.json')
   if existing.exists():
    record=json.loads(existing.read_text())
    assert record['arm']==arm and len(record['seconds'])==20 and record['warmup']==10
    print('Reusing completed benchmark',existing,flush=True)
    continue
   call(['/workspace/envs/train/bin/python','scripts/camera_ready/benchmark.py','--arm',arm,'--warmup','10','--updates','20','--output',f'results/camera-ready/benchmark-{arm}-{repeat}.json'],f'logs/camera-ready/benchmark-{arm}-{repeat}.log',env)
 state('remaining_evaluation')
 paired(evaluate,([1337,1339],0,'remaining'),([1338],1,'remaining'))
 state('complete')
if __name__=='__main__':
 try:main()
 except Exception:
  state('failed',error=traceback.format_exc());raise
