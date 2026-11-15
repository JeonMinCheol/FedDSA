"""Current-state, source-class local marginal utility; no server EMA or weights.
Computes an exact virtual weight derivative for a frozen backbone, linear
projector/classifier and fixed current PCW. Real SGD remains untouched.
"""
import torch
import torch.nn.functional as F
from utils.mcl_gap import add_mcl_gap
from utils.distance_numerics import guarded_cdist


def support_terms(x,y,projector,head,anchors,pcw,lamda,mcl,gap,class_cap):
    z=F.linear(x,projector.weight,projector.bias)
    ce=F.cross_entropy(F.linear(z,head.weight,head.bias),y)
    distance=guarded_cdist(z,anchors)
    distance=add_mcl_gap(distance,y,anchors,gap,class_cap=class_cap)
    per=lamda*F.mse_loss(z,anchors[y],reduction='none').mean(dim=1)+mcl*F.cross_entropy(-distance,y,reduction='none')
    return z,ce,per


def local_marginal_gate(sx,sy,qx,qy,projector,head,anchors,pcw,lr,lamda,mcl,gap,class_cap):
    params=[projector.weight,projector.bias,head.weight,head.bias]
    if any(p is None for p in params):raise ValueError('Biased linear projector and classifier required')
    sx=sx.detach();qx=qx.detach();anchors=anchors.detach();pcw=pcw.detach().to(sx)
    z,ce,per=support_terms(sx,sy,projector,head,anchors,pcw,lamda,mcl,gap,class_cap)
    r=torch.autograd.grad(per.sum(),z,retain_graph=True)[0].detach()
    grads=torch.autograd.grad(ce+(per*pcw).mean(),params)
    updated=[(p-lr*g).detach().requires_grad_() for p,g in zip(params,grads)]
    query_ce=F.cross_entropy(F.linear(F.linear(qx,updated[0],updated[1]),updated[2],updated[3]),qy)
    v_w,v_b=torch.autograd.grad(query_ce,updated[:2])
    # -d L_query/d w_c = lr * <g_query_projector, g_anchor_source_class_c>.
    contribution=(r*F.linear(sx,v_w,v_b)).sum(dim=1)*pcw/len(sy)
    score=torch.zeros(len(anchors),device=sx.device,dtype=sx.dtype)
    score.index_add_(0,sy,lr*contribution)
    # Exact source-class gradient norms using a batch Gram matrix; avoids C*D*D storage.
    weighted_r=r*pcw[:,None]
    gram=(weighted_r@weighted_r.T)*(sx@sx.T+1.)
    same=sy[:,None]==sy[None,:]
    square=torch.zeros_like(score)
    square.index_add_(0,sy,(gram*same).sum(dim=1)/len(sy)**2)
    a_norm=square.clamp_min(0).sqrt()
    q_norm=(v_w.square().sum()+v_b.square().sum()).sqrt()
    denominator=lr*a_norm*q_norm
    observed=torch.bincount(sy,minlength=len(anchors))>0
    usable=observed & (denominator>torch.finfo(sx.dtype).tiny)
    cosine=torch.zeros_like(score)
    cosine[usable]=(score[usable]/denominator[usable]).clamp(-1,1)
    weight=1.+.25*cosine
    if not all(bool(torch.isfinite(v).all()) for v in [score,cosine,weight]):
        raise FloatingPointError('Nonfinite current local marginal gate')
    return {'weight':weight.detach(),'score':score.detach(),'cosine':cosine.detach(),
        'observed':observed,'support_count':torch.bincount(sy,minlength=len(anchors)),
        'anchor_gradient_norm':a_norm.detach(),'query_gradient_norm':q_norm.detach(),
        'query_loss':query_ce.detach()}


def evaluate_gate(sx,sy,dx,dy,projector,head,anchors,pcw,weights,lr,lamda,mcl,actual_gap,class_cap):
    """Observer B3: evaluate both arms using the ACTUAL local loss, including gap.
    These losses never select a gate or update real parameters.
    """
    params=[projector.weight,projector.bias,head.weight,head.bias]
    _,ce,per=support_terms(sx.detach(),sy,projector,head,anchors.detach(),pcw,lamda,mcl,actual_gap,class_cap)
    losses={}
    for name,w in [('uniform',torch.ones_like(weights)),('local_gate',weights)]:
        grads=torch.autograd.grad(ce+(per*pcw.detach()*w[sy]).mean(),params,retain_graph=name=='uniform')
        updated=[(p-lr*g).detach() for p,g in zip(params,grads)]
        with torch.no_grad():
            logits=F.linear(F.linear(dx.detach(),updated[0],updated[1]),updated[2],updated[3])
            losses[name]=F.cross_entropy(logits,dy,reduction='none')
    gain=losses['uniform']-losses['local_gate']
    return {'query_batch':3,'validation_gap_included':True,'gain_over_uniform_mean':float(gain.mean()),
            'query_samples':int(len(dy)),'query_labels':dy.cpu().tolist(),'gain_over_uniform':gain.cpu().tolist(),
            'feeds_training':False,'scope':'Disjoint within this pre-SGD probe, not a permanently withheld training split'}
