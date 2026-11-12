"""Separate finite-cohort heterogeneity, measured query noise and temporal drift."""
import math

import torch


@torch.no_grad()
def cohort_candidate_confidence(scores, valid, query_variance, client_weights,
                                population_clients, previous_state=None, ema=.25):
    """Experimental confidence estimator; no running experiment uses this yet.

    All observed client means are fixed strata when participation is complete.
    Their heterogeneity is not itself query-measurement error. Negative effects
    remain an explicit downside penalty, even when their signs are certain.
    Partial participation retains a conservative finite-cohort sampling term.
    The two-query-half variance is conditional on the virtual support batch;
    this is not a calibrated confidence interval for all training uncertainty.
    """
    if scores.ndim!=2 or scores.shape!=valid.shape or scores.shape!=query_variance.shape:
        raise ValueError('Scores, masks and query variances need client by class shape')
    if client_weights.shape!=(scores.size(0),) or scores.size(1)==0:
        raise ValueError('One weight is required per client and at least one class')
    if not isinstance(population_clients,int) or population_clients<max(1,scores.size(0)):
        raise ValueError('Population must include all observed clients')
    if not math.isfinite(ema) or not 0<ema<=1:
        raise ValueError('EMA must be in (0,1]')
    values=scores.double();mask=valid.bool();variance=query_variance.to(values)
    weights=client_weights.to(values)
    if not torch.isfinite(values[mask]).all() or bool((values[mask].abs()>1+1e-6).any()):
        raise ValueError('Valid scores must be finite and in [-1,1]')
    if not torch.isfinite(variance[mask]).all() or bool((variance[mask]<0).any()):
        raise ValueError('Measured query variances must be finite and nonnegative')
    if not torch.isfinite(weights).all() or bool((weights<0).any()):
        raise ValueError('Client weights must be finite and nonnegative')
    weighted=mask.to(values)*weights[:,None]
    total=weighted.sum(dim=0);present=total>0
    shares=weighted/total.clamp_min(1e-30)
    safe=torch.where(mask,values,torch.zeros_like(values))
    safe_variance=torch.where(mask,variance,torch.zeros_like(variance))
    mean=(shares*safe).sum(dim=0)
    downside=(shares*(-safe).clamp_min(0)).sum(dim=0)
    count=total.square()/weighted.square().sum(dim=0).clamp_min(1e-30)
    observed=(mask & (weights[:,None]>0)).sum(dim=0)
    fraction=observed.to(values)/population_clients
    within=(shares.square()*safe_variance).sum(dim=0)
    between=(shares*(safe-mean[None,:]).square()).sum(dim=0)
    between=torch.where(count>1,between*count/(count-1).clamp_min(1e-30),torch.ones_like(between))
    sampling=(1-fraction)*between/count.clamp_min(1)
    current={'mean':mean,'round_mean_second':mean.square(),'noise_variance':within+sampling,
             'negative_mass':downside,'count':count}
    state={}
    old_count=previous_state['count'].to(count) if previous_state is not None else torch.zeros_like(count)
    seen=present & (old_count>0)
    for name,value in current.items():
        old=previous_state[name].to(value) if previous_state is not None else torch.zeros_like(value)
        updated=torch.where(present,value,old)
        updated[seen]=(1-ema)*old[seen]+ema*value[seen]
        state[name]=updated
    effective=torch.minimum(state['count'],count)
    temporal=(state['round_mean_second']-state['mean'].square()).clamp_min(0)
    # A newly noisy batch cannot inherit an artificially small uncertainty
    # from a quiet EMA history. Keep the current measurement as a floor.
    error=(torch.maximum(state['noise_variance'],current['noise_variance'])+temporal).clamp_min(0).sqrt()
    positive=state['mean'].clamp_min(0)
    confidence=(effective/(effective+1))*positive/(positive+error+state['negative_mass']+1e-12)
    confidence=torch.where(present,confidence,torch.zeros_like(confidence))
    return {'mean':state['mean'],'standard_error':error,'negative_mass':state['negative_mass'],
            'current_mean':mean,'current_standard_error':current['noise_variance'].sqrt(),
            'current_negative_mass':downside,
            'effective_clients':effective,'confidence':confidence,'state':state,
            'query_mean_variance':within,'client_sampling_variance':sampling,
            'temporal_mean_variance':temporal,'observed_population_fraction':fraction,
            'scope':'Experimental conditional query-noise and drift estimate, not a calibrated full-training interval'}
