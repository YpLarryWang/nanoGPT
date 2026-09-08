#!/usr/bin/env python3
import argparse,importlib.metadata,json,platform,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['train','eval'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
expected=Path('config/camera_ready')/f'{a.kind}-requirements.txt'
versions={};mismatch={}
for line in expected.read_text().splitlines():
 if '==' not in line:continue
 name,version=line.split('==',1)
 try:actual=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:actual=None
 versions[name]=actual
 if actual!=version:mismatch[name]={'expected':version,'actual':actual}
import torch
report={'kind':a.kind,'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
 'cuda_available':torch.cuda.is_available(),'gpu_count':torch.cuda.device_count(),'versions':versions,'mismatch':mismatch,
 'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,name,uuid,power.limit','--format=csv'],text=True),
 'note':'Dependency snapshot from July 19 Vast2 W&B backup; host driver/OS may differ.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
assert not mismatch,mismatch
assert platform.python_version()=='3.12.13'
assert torch.cuda.is_available()
print('Exact pinned dependencies and Python match:',a.kind,torch.__version__,torch.version.cuda)
