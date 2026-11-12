"""Repair only cancellation-prone Euclidean distances; record passive traces.

The untouched MM result/graph is returned outside its rounding-error region.
No gradient, beta, prototype, model parameter, learning rate or RNG is altered
by diagnostics. A demonstration of the numeric defect is not proof that it was
the first cause of the historical local295 failure; the rerun tests that.
"""
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F


def guarded_cdist(left, right, *, self_pairs=False, keep_double_fallback=False, audit=None, context='distance'):
    if (left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]
            or left.dtype != right.dtype or left.device != right.device
            or left.dtype not in (torch.float32,torch.float64)):
        raise ValueError('Matching float32/float64 Euclidean matrices required')
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise FloatingPointError('Nonfinite distance inputs')
    original = torch.cdist(left,right,p=2)
    with torch.no_grad():
        x2=left.detach().double().square().sum(1)
        a2=right.detach().double().square().sum(1)
        # Conservative gamma_(D+3) dot/norm/addition error bound. This is a
        # floating point conditioning test, not a fitted statistical threshold.
        k=left.shape[1]+3;eps=torch.finfo(left.dtype).eps
        gamma=(k*eps)/(1-k*eps)
        bound=gamma*(x2[:,None]+a2[None,:]+2*x2.sqrt()[:,None]*a2.sqrt()[None,:])
        eligible=torch.ones_like(original,dtype=torch.bool)
        if self_pairs:
            if left.shape!=right.shape or not torch.equal(left.detach(),right.detach()):
                raise ValueError('Self-distance diagonal exclusion requires the same bank')
            eligible.fill_diagonal_(False)
        suspect=eligible&((original.detach().double().square()<=bound)|~torch.isfinite(original.detach()))
        count=int(suspect.sum())
        meta={'fallback':count>0,'suspect_pairs':count,'eligible_pairs':int(eligible.sum()),
            'max_squared_roundoff_bound':float(bound.max()) if bound.numel() else 0.,
            'original_nonfinite':int((~torch.isfinite(original.detach())&eligible).sum()),
            'self_diagonal_excluded':bool(self_pairs),'dimension':int(left.shape[1])}
    if count:
        # Drop the original graph entirely: even a masked 0*NaN derivative
        # from a damaged MM graph must not contaminate a finite direct result.
        direct=torch.cdist(left.double(),right.double(),p=2,compute_mode='donot_use_mm_for_euclid_dist')
        result=direct if keep_double_fallback else direct.to(left)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('Direct Euclidean fallback is not finite')
    else:
        result=original
    if audit is not None:audit.distance(context,meta)
    return result


def nearest_gap(bank,audit=None):
    distances=guarded_cdist(bank,bank,self_pairs=True,keep_double_fallback=True,
        audit=audit,context='nearest_gap')
    diagonal=torch.eye(bank.shape[0],device=bank.device,dtype=torch.bool)
    nearest=distances.masked_fill(diagonal,float('inf')).min(1).values
    nearest=nearest[torch.isfinite(nearest)]
    return float(nearest.mean().to(bank).item()) if nearest.numel() else 0.


def _write(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a')as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')


def _norms(model,gradients=False):
    groups={'backbone':[],'projector':[],'head':[]}
    for name,param in model.named_parameters():
        value=param.grad if gradients else param.detach()
        if value is None:continue
        key='projector'if name.startswith('shared_projector.')else 'head'if name.startswith('shared_head.')else 'backbone'
        groups[key].append(value.detach().double().norm())
    result={k:float(torch.stack(v).norm())if v else 0. for k,v in groups.items()}
    result['total']=math.hypot(*result.values())
    return result


def _bn(model):
    means=[];variances=[];gammas=[];nonfinite=0
    for module in model.modules():
        if not isinstance(module,torch.nn.modules.batchnorm._BatchNorm):continue
        if module.running_mean is not None:
            means.append(float(module.running_mean.detach().abs().max()))
            variances.append(float(module.running_var.detach().min()))
            nonfinite+=int(not bool(torch.isfinite(module.running_mean).all()&torch.isfinite(module.running_var).all()))
        if module.weight is not None:gammas.append(float(module.weight.detach().abs().max()))
    return {'layers':len(means),'running_mean_absmax':max(means,default=0.),
        'running_var_min':min(variances,default=0.),'gamma_absmax':max(gammas,default=0.),'nonfinite_layers':nonfinite}


class LocalDistanceAudit:
    """All-client per-round routing; suspect-client15 per-batch finite traces."""
    def __init__(self,client_id,seed,goal):
        self.client=int(client_id);self.seed=int(seed);self.goal=str(goal)
        self.round=-1;self.calls={};self.weights=None;self.batch=0;self.eval_norm_max=0.;self.closed=True
        self.directory=Path(__file__).resolve().parent.parent/'logs'

    def begin(self,client,model):
        self.round=int(client.current_round);self.calls={};self.weights=None;self.batch=0;self.eval_norm_max=0.;self.closed=False
        self.before={'parameters':_norms(model),'BN':_bn(model)}if self.client==15 else None

    def distance(self,context,meta):
        if meta['fallback']:
            _write(self.directory/('nextgap_distance_fallbacks_'+self.goal+'.jsonl'),
                dict(round=self.round,client=self.client,seed=self.seed,goal=self.goal,context=context,
                     after_train_audit_closed=self.closed,**meta))
        row=self.calls.setdefault(context,dict(calls=0,fallback_calls=0,suspect_pairs=0,original_nonfinite=0,max_squared_roundoff_bound=0.))
        row['calls']+=1;row['fallback_calls']+=int(meta['fallback']);row['suspect_pairs']+=meta['suspect_pairs']
        row['original_nonfinite']+=meta['original_nonfinite'];row['max_squared_roundoff_bound']=max(row['max_squared_roundoff_bound'],meta['max_squared_roundoff_bound'])

    def retain(self,representation):
        if self.client==15 and representation.requires_grad:representation.retain_grad()

    @torch.no_grad()
    def batch_gradient(self,client,model,representation,logits,labels,task,anchor):
        if self.client!=15:return
        gradients=_norms(model,gradients=True)
        if not math.isfinite(gradients['total']):
            raise FloatingPointError('Nonfinite local gradient before optimizer step')
        row={'round':self.round,'client':self.client,'seed':self.seed,'goal':self.goal,'batch':self.batch,
            'task_loss':float(task.detach()),'anchor_loss':float(anchor.detach()),'gradient_L2':gradients,
            'representation_max_L2':float(representation.detach().double().norm(dim=1).max()),
            'head_weight_L2':float(model.shared_head.weight.detach().double().norm()),
            'projector_weight_L2':float(model.shared_projector.weight.detach().double().norm()),
            'before_optimizer_step':True,'gradient_clipping':False,'feeds_update':False}
        if representation.grad is not None and self.weights is not None:
            count=labels.numel();errors=logits.detach().softmax(1)
            errors=errors-F.one_hot(labels,logits.shape[1]).to(errors)
            task_grad=errors@model.shared_head.weight.detach()/count
            mse_grad=2*float(client.lamda)*self.weights[:,None]*(representation.detach()-client.semantic_anchors[labels].to(representation))/(count*representation.shape[1])
            residual=representation.grad.detach()-task_grad-mse_grad
            bound=2*float(client.mcl)*self.weights/count
            tolerance=128*torch.finfo(representation.dtype).eps*(1+task_grad.norm(dim=1)+mse_grad.norm(dim=1))
            excess=(residual.norm(dim=1)-bound-tolerance).clamp_min(0)
            row.update(MCL_rep_gradient_bound_violations=int((excess>0).sum()),
                MCL_rep_gradient_max_excess=float(excess.max()),
                bound_scope='Frozen anchors, mean CE, detached agreement/utility weights; task+MSE subtracted with an F32 rounding allowance.')
        _write(self.directory/('nextgap_local_numeric_batches_'+self.goal+'.jsonl'),row)
        self.batch+=1

    @torch.no_grad()
    def eval_rep(self,representation):
        if self.client==15:self.eval_norm_max=max(self.eval_norm_max,float(representation.detach().double().norm(dim=1).max()))

    def finish(self,client,model):
        row={'round':self.round,'seed':self.seed,'client':self.client,'goal':self.goal,
            'distance_contexts':self.calls,'batch_trace_count':self.batch,'feeds_update':False,
            'learning_rate':float(client.learning_rate),'gradient_clipping':False,
            'prototype_gap_after':(getattr(client,'anchor_candidate_evidence',None)or{}).get('next_local_gap'),
            'prototype_norm_max':max((float(v.double().norm())for v in (client.protos or{}).values()),default=0.)}
        if self.client==15:row.update(before=self.before,after={'parameters':_norms(model),'BN':_bn(model)},eval_feature_norm_max=self.eval_norm_max)
        _write(self.directory/('nextgap_distance_numerics_'+self.goal+'.jsonl'),row)
        self.weights=None;self.closed=True
