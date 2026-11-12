"""Experimental refinement of RAAR's original beta, retaining its stable return.

Unknown directional evidence preserves the original geometry beta. This is a
heuristic confidence update, not a calibrated posterior or class normalization.
Training callers must validate proposals before applying this experimental rule.
"""
import math

import torch


def _strength(mean,error,downside,count):
    present=count>0
    support=count/(count+1)
    positive=support*(mean-2*error-downside).clamp_min(0)/(
        mean.clamp_min(0)+error+downside+1e-12)
    negative=support*(-mean-2*error).clamp_min(0)/(
        (-mean).clamp_min(0)+error+1e-12)
    return torch.where(present,positive,0),torch.where(present,negative,0)


@torch.no_grad()
def propose_raar_beta(prior_beta,estimates,max_weight=1.5):
    """Propose class changes without erasing RAAR when the query is uncertain.

    Estimates must describe the actual stable-to-EMA-candidate path, rather
    than the current-to-candidate path used by the older candidate-gain mode.
    Positive evidence must persist in both EMA and current measurements;
    credible adverse current evidence can suppress immediately.
    A proposal is not an accepted update: finite independent query validation
    against the original beta is required before applying it.
    """
    if prior_beta.ndim!=1 or not prior_beta.numel() or not prior_beta.is_floating_point():
        raise ValueError('A floating geometry beta is required for each class')
    if not torch.isfinite(prior_beta).all() or (prior_beta<0).any() or (prior_beta>1).any():
        raise ValueError('Original RAAR beta must be finite and within [0,1]')
    if len(estimates)!=3 or not math.isfinite(max_weight) or max_weight<=1:
        raise ValueError('Three path intervals and a maximum above one are required')
    prior=prior_beta.double()
    favorable=[];adverse=[]
    for estimate in estimates:
        def value(name,default=None):
            result=estimate.get(name,default)
            if result is None or result.shape!=prior.shape:
                raise ValueError(f'{name} needs one value per anchor class')
            result=result.to(prior)
            if not torch.isfinite(result).all():
                raise ValueError(f'{name} must be finite')
            return result
        mean=value('mean');error=value('standard_error');count=value('effective_clients')
        downside=value('negative_mass',torch.zeros_like(prior))
        if (error<0).any() or (count<0).any() or (downside<0).any():
            raise ValueError('Errors, client support and downside must be nonnegative')
        positive,negative=_strength(mean,error,downside,count)
        if 'current_mean' in estimate:
            current_mean=value('current_mean');current_error=value('current_standard_error')
            current_downside=value('current_negative_mass',torch.zeros_like(prior))
            if (current_error<0).any() or (current_downside<0).any():
                raise ValueError('Current errors and downside must be nonnegative')
            current_positive,current_negative=_strength(current_mean,current_error,current_downside,count)
            positive=torch.minimum(positive,current_positive)
            negative=torch.maximum(negative,current_negative)
        favorable.append(positive);adverse.append(negative)
    # This is a per-class inverse of .5*r*(1+r), not a mean normalization.
    interval_prior=(torch.sqrt(1+8*prior)-1)/2
    first=interval_prior+(1-interval_prior)*favorable[0]-interval_prior*adverse[0]
    second=interval_prior+(1-interval_prior)*favorable[1]-interval_prior*adverse[1]
    base=.5*first+.5*first*second
    extra=(float(max_weight)-1)*first*second*favorable[0]*favorable[1]*favorable[2]
    beta=base+extra
    unchanged=(favorable[0]==0)&(favorable[1]==0)&(adverse[0]==0)&(adverse[1]==0)
    beta=torch.where(unchanged,prior,beta)
    return {'beta':beta.to(prior_beta),'favorable_evidence':torch.stack(favorable),
        'adverse_evidence':torch.stack(adverse),'summary':{
            'mode':'raar_geometry_prior_refinement','application_reference':'slow_stable_anchor',
            'uncertain_evidence_preserves_original_beta':True,
            'class_mean_normalization':False,'coefficient_one_is_fixed_ema':True,
            'coefficient_zero_is_stable_anchor':True,'stable_return_retained':True,
            'maximum_beta':float(max_weight),'mean_beta':float(beta.mean()),
            'raised_classes':int((beta>prior+1e-8).sum()),
            'lowered_classes':int((beta<prior-1e-8).sum()),
            'unchanged_classes':int(unchanged.sum()),
            'mean_extra_interval_increment':float(extra.mean()),
            'score_path_required':'stable anchor to actual EMA candidate',
            'proposal_only':True,'finite_independent_validation_required':True,
            'calibrated_probability':False}}


@torch.no_grad()
def validate_raar_proposal(prior_beta,proposal_beta,evaluate,max_weight=1.5):
    """Backtrack the beta adjustment using finite paired query gains.

    ``evaluate(trial, reference)`` must use independent cached training queries
    and return mean_gain, standard_error and observed_clients. No test data or
    gradient-sign-only acceptance is allowed. Only the beta adjustment is
    backtracked; the fallback is the original complete RAAR update.
    """
    if prior_beta.ndim!=1 or proposal_beta.shape!=prior_beta.shape:
        raise ValueError('Prior and proposed beta shapes must match')
    if (not prior_beta.is_floating_point() or proposal_beta.dtype!=prior_beta.dtype
            or proposal_beta.device!=prior_beta.device):
        raise ValueError('Prior and proposed beta must share floating dtype and device')
    if not math.isfinite(max_weight) or max_weight<=1:
        raise ValueError('The maximum coefficient must exceed one')
    if not all(bool(torch.isfinite(v).all()) for v in (prior_beta,proposal_beta)):
        raise ValueError('Beta values must be finite')
    if (prior_beta<0).any() or (prior_beta>1).any() or (proposal_beta<0).any() or (proposal_beta>max_weight+1e-6).any():
        raise ValueError('Beta values exceed their admissible ranges')
    trace=[]
    def improves(trial,reference,kind,factor):
        result=evaluate(trial,reference)
        valid=(isinstance(result,dict) and result.get('observed_clients',0)>0
            and isinstance(result.get('mean_gain'),(int,float))
            and isinstance(result.get('standard_error'),(int,float)))
        floor=result.get('numerical_tolerance',0.) if isinstance(result,dict) else 0.
        valid=valid and math.isfinite(result['mean_gain']) and math.isfinite(result['standard_error']) and result['standard_error']>=0
        valid=valid and isinstance(floor,(int,float)) and math.isfinite(floor) and floor>=0
        accepted=bool(valid and result['mean_gain']-2*result['standard_error']>floor)
        trace.append({'factor':factor,'comparison':kind,'valid':bool(valid),
            'accepted':accepted,'mean_gain':result.get('mean_gain') if valid else None,
            'standard_error':result.get('standard_error') if valid else None,
            'numerical_tolerance':floor if valid else None})
        return accepted
    if not torch.equal(prior_beta,proposal_beta):
        for factor in (1.,.5,.25,.125,.0625,.03125):
            trial=prior_beta+factor*(proposal_beta-prior_beta)
            if torch.equal(trial,prior_beta):
                break
            if not improves(trial,prior_beta,'finite_gain_over_original_raar',factor):
                continue
            if bool((trial>1).any()):
                prefix=trial.clamp_max(1)
                if not torch.equal(prefix,prior_beta) and not improves(
                        prefix,prior_beta,'gain_before_extrapolation',factor):
                    continue
                if not improves(trial,prefix,'incremental_gain_beyond_fixed_ema',factor):
                    continue
            return {'beta':trial,'accepted_factor':factor,'fallback_to_original_beta':False,
                    'comparisons':trace,'physical_step_clipping':False}
    return {'beta':prior_beta.clone(),'accepted_factor':None,'fallback_to_original_beta':True,
            'comparisons':trace,'physical_step_clipping':False}


@torch.no_grad()
def apply_raar_beta(current,ema_candidate,stable,beta,prototype_gamma):
    """Apply S + beta*(C-S), with C already containing the prototype EMA."""
    if current.ndim!=2 or current.shape!=ema_candidate.shape or current.shape!=stable.shape:
        raise ValueError('Anchor banks must have matching class by feature shapes')
    if any(value.device!=current.device or value.dtype!=current.dtype for value in (ema_candidate,stable)):
        raise ValueError('Anchor banks must have matching dtype and device')
    if beta.shape!=current.shape[:1] or not math.isfinite(prototype_gamma) or not 0<=prototype_gamma<=1:
        raise ValueError('One beta per class and a valid prototype EMA weight are required')
    if not all(bool(torch.isfinite(value).all()) for value in (current,ema_candidate,stable,beta)) or (beta<0).any():
        raise ValueError('Anchor banks and nonnegative coefficients must be finite')
    weights=beta.to(current)
    updated=stable+weights[:,None]*(ema_candidate-stable)
    updated=torch.where((weights==1)[:,None],ema_candidate,updated)
    forward=weights[:,None]*(ema_candidate-current)
    returned=(1-weights)[:,None]*(stable-current)
    error=(updated-current-forward-returned).norm(dim=1)
    return updated,{'applied_beta':weights,'effective_prototype_weight':float(prototype_gamma)*weights,
        'stable_coefficient':1-weights,'forward_component':forward,'return_component':returned,
        'decomposition_max_abs_l2_error':float(error.max()),
        'coefficient_one_is_fixed_ema':True,'stable_return_retained':True,
        'physical_step_clipping':False,'class_mean_normalization':False}


@torch.no_grad()
def diagnose_raar_beta_adjustments(prior_beta,beta,evaluate):
    """Measure suppression, amplification and their interaction on batch4.

    ``evaluate`` must compare finite losses on the same cached diagnostic
    queries. This observer never selects an update. In particular a positive
    combined gain does not establish that both adjustment groups are useful.
    """
    if (prior_beta.ndim!=1 or not prior_beta.numel() or beta.shape!=prior_beta.shape
            or not prior_beta.is_floating_point() or beta.dtype!=prior_beta.dtype
            or beta.device!=prior_beta.device):
        raise ValueError('Aligned floating beta vectors on the same device are required')
    if (not bool(torch.isfinite(prior_beta).all()) or not bool(torch.isfinite(beta).all())
            or bool((prior_beta<0).any()) or bool((prior_beta>1).any()) or bool((beta<0).any())):
        raise ValueError('Original beta must be within [0,1] and applied beta nonnegative')
    lower=torch.minimum(prior_beta,beta)
    higher=torch.maximum(prior_beta,beta)
    comparisons=(('combined',beta,prior_beta),('suppression_only',lower,prior_beta),
        ('amplification_only',higher,prior_beta),
        ('suppression_after_amplification',beta,higher),
        ('amplification_after_suppression',beta,lower))
    result={'available':False,'feeds_update':False,'query_batch':4,
        'reference':'same_state_original_RAAR_beta',
        'suppressed_classes':int((beta<prior_beta).sum()),
        'raised_classes':int((beta>prior_beta).sum()),'contrasts':{},
        'interaction_standard_error':None,
        'scope':'Same cached training queries; group effects may interact across anchor classes. This is not a full-training or independent-seed result.'}
    memo={}
    signature=None
    for name,trial,reference in comparisons:
        key=(tuple(trial.double().cpu().tolist()),tuple(reference.double().cpu().tolist()))
        if key not in memo:
            memo[key]=evaluate(trial.clone(),reference.clone())
        measurement=memo[key]
        result['comparison_evaluations']=len(memo)
        if not isinstance(measurement,dict) or measurement.get('observed_clients',0)<=0:
            result['reason']='Incomplete diagnostic queries'
            return result
        if (measurement.get('query_batch')!=4 or measurement.get('uses_test_data') is not False
                or measurement.get('model_parameters_changed') is not False):
            result['reason']='A diagnostic must use batch4 without test data or model changes'
            return result
        gain=measurement.get('mean_gain');error=measurement.get('standard_error')
        count=measurement.get('query_samples',0)
        if (not isinstance(gain,(int,float)) or not math.isfinite(gain)
                or not isinstance(error,(int,float)) or not math.isfinite(error) or error<0
                or not isinstance(count,int) or count<=0):
            result['reason']='Invalid finite paired-query measurement'
            return result
        current_signature=(measurement['observed_clients'],count)
        if signature is None:signature=current_signature
        if current_signature!=signature:
            result['reason']='All contrasts must use the same cached diagnostic cohort'
            return result
        result['contrasts'][name]=dict(measurement)
    gains={key:value['mean_gain'] for key,value in result['contrasts'].items()}
    closure=max(abs(gains['combined']-gains['suppression_only']-
                    gains['amplification_after_suppression']),
                abs(gains['combined']-gains['amplification_only']-
                    gains['suppression_after_amplification']))
    result['paired_gain_closure_max_abs_error']=closure
    if closure>1e-10*max(1.,max(abs(value) for value in gains.values())):
        result['reason']='Contrasts do not close on a common finite loss reference'
        return result
    result.update(available=True,observed_clients=signature[0],query_samples=signature[1],
        interaction_mean_gain=gains['combined']-gains['suppression_only']-gains['amplification_only'],
        interaction_uncertainty='Not estimated: the contrast errors are correlated and must not be added as independent errors')
    return result


@torch.no_grad()
def compare_raar_movement(current,ema_candidate,stable,prior_beta,beta,prototype_gamma):
    """Compare original and applied RAAR movement at the exact same state."""
    if (prior_beta.shape!=beta.shape or not prior_beta.is_floating_point()
            or prior_beta.dtype!=beta.dtype or prior_beta.device!=beta.device
            or bool((prior_beta<0).any()) or bool((prior_beta>1).any())):
        raise ValueError('The reference must be the aligned original RAAR beta in [0,1]')
    reference,old=apply_raar_beta(current,ema_candidate,stable,prior_beta,prototype_gamma)
    updated,new=apply_raar_beta(current,ema_candidate,stable,beta,prototype_gamma)
    def lengths(value):return value.double().norm(dim=1)
    old_step=lengths(reference-current);new_step=lengths(updated-current)
    lower=beta<prior_beta;higher=beta>prior_beta
    tolerance=16*torch.finfo(current.dtype).eps*torch.maximum(
        lengths(current),torch.maximum(lengths(stable),lengths(ema_candidate))).clamp_min(1)
    smaller=new_step<old_step-tolerance
    larger=new_step>old_step+tolerance
    def values(tensor):return tensor.detach().cpu().tolist()
    def summary(mask):
        if not bool(mask.any()):return {'classes':0}
        return {'classes':int(mask.sum()),
            'reference_mean_prototype_component_l2':float(lengths(old['forward_component'])[mask].mean()),
            'applied_mean_prototype_component_l2':float(lengths(new['forward_component'])[mask].mean()),
            'reference_mean_stable_component_l2':float(lengths(old['return_component'])[mask].mean()),
            'applied_mean_stable_component_l2':float(lengths(new['return_component'])[mask].mean()),
            'reference_mean_actual_step_l2':float(old_step[mask].mean()),
            'applied_mean_actual_step_l2':float(new_step[mask].mean()),
            'smaller_actual_step_classes':int((mask&smaller).sum()),
            'larger_actual_step_classes':int((mask&larger).sum())}
    return {'reference':'same_state_original_RAAR_beta',
        'candidate_prototype_weight':float(prototype_gamma),'physical_step_clipping':False,
        'class_mean_normalization':False,'coefficient_one_is_fixed_ema':True,
        'lowered_beta':summary(lower),'raised_beta':summary(higher),
        'unchanged_beta':summary(~(lower|higher)),
        'all_classes':summary(torch.ones_like(lower)),
        'outward_stable_component_classes':int((beta>1).sum()),
        'classwise':{'prior_beta':values(prior_beta),'applied_beta':values(beta),
            'reference_prototype_component_l2':values(lengths(old['forward_component'])),
            'applied_prototype_component_l2':values(lengths(new['forward_component'])),
            'reference_stable_component_l2':values(lengths(old['return_component'])),
            'applied_stable_component_l2':values(lengths(new['return_component'])),
            'reference_actual_step_l2':values(old_step),'applied_actual_step_l2':values(new_step),
            'reference_stable_coefficient':values(old['stable_coefficient']),
            'applied_stable_coefficient':values(new['stable_coefficient'])},
        'interpretation':'A beta reduction suppresses the prototype component but can enlarge the stable component and actual movement. These norms alone do not establish noise removal.'}
