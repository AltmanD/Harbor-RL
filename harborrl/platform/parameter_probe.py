"""Observer-only optimizer hook: sample actual parameter tensors before/after step."""
import json
from pathlib import Path
import torch
import os
def before_step(args,rollout_id,step_id,model,optimizer,scheduler):
 optimizer._harbor_probe_step=(rollout_id,step_id)
 if getattr(optimizer,'_harbor_probe_installed',False):return
 optimizer._harbor_probe_installed=True
 original=optimizer.step
 selected=[]
 for chunk in model:
  for name,p in chunk.named_parameters():
   if p.requires_grad and ('linear_qkv.weight' in name or 'linear_fc1.weight' in name):
    selected.append((name,p))
    if len(selected)>=3:break
  if len(selected)>=3:break
 def step(*a,**kw):
  before=[p.detach().reshape(-1)[:65536].float().clone() for _,p in selected]
  result=original(*a,**kw)
  rank=torch.distributed.get_rank(); records=[]
  for (name,p),old in zip(selected,before):
   new=p.detach().reshape(-1)[:65536].float();delta=new-old
   records.append(dict(name=name,numel=old.numel(),delta_l2=delta.norm().item(),delta_abs_max=delta.abs().max().item(),changed=int((delta!=0).sum().item()),finite=bool(torch.isfinite(new).all().item())))
  r,s=optimizer._harbor_probe_step
  out=dict(rollout_id=r,step_id=s,rank=rank,update_successful=bool(result[0]),grad_norm=float(result[1]),parameters=records)
  path=Path(os.environ['RUN_DIR'])/'manifests'/f'parameter-updates-rank{rank}.jsonl'
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('a') as f:f.write(json.dumps(out)+'\n')
  return result
 optimizer.step=step
