"""Estimate whether each proposed anchor direction improves held-out query loss."""
import math

import torch
import torch.nn.functional as F
from utils.mcl_gap import add_mcl_gap


def anchor_coordinate_direction_scores(gradient, direction):
    """Keep relative anchor-gradient magnitudes within each client's vote.

    A separate cosine for every anchor gives a negligible derivative the same
    vote as an informative one. A single bank-gradient norm bounds a client's
    influence while preserving the relative magnitude of its class evidence.
    This does not normalize applied class coefficients or movement budgets.
    """
    if gradient.ndim != 2 or gradient.shape != direction.shape:
        raise ValueError('Anchor gradient and displacement matrices must match')
    gradient = gradient.double()
    direction = direction.double()
    gradient_norm = gradient.norm(dim=1)
    direction_norm = direction.norm(dim=1)
    bank_norm = gradient.norm()
    valid = (gradient_norm > 0) & (direction_norm > 0)
    numerator = -(gradient * direction).sum(dim=1)
    scores = torch.zeros_like(numerator)
    class_cosine = torch.zeros_like(numerator)
    relative_norm = torch.zeros_like(numerator)
    if bool(valid.any()):
        scores[valid] = (numerator[valid] / (bank_norm * direction_norm[valid])).clamp(-1, 1)
        class_cosine[valid] = (numerator[valid] / (gradient_norm[valid] * direction_norm[valid])).clamp(-1, 1)
    if bool(bank_norm > 0):
        relative_norm = gradient_norm / bank_norm
    return {'scores': scores, 'valid': valid,
            'class_cosine': class_cosine, 'relative_gradient_norm': relative_norm}


def virtual_support_update(support_features, support_labels,
                       projector, head, anchors, inner_lr, mse_weight, mcl_weight,
                       mcl_gap, create_graph=False, mcl_gap_class_cap=False,
                       next_gap_context=None):
    """Prepare one differentiable support update for independent query losses."""
    if not isinstance(projector,torch.nn.Linear) or not isinstance(head,torch.nn.Linear):
        raise ValueError('Candidate trust requires a linear projector and classifier')
    if projector.bias is None:
        raise ValueError('Candidate trust requires a biased projector')
    with torch.enable_grad():
        if next_gap_context is not None:
            from utils.raar_next_gap import resolve_next_gap
            mcl_gap=resolve_next_gap(anchors,mcl_gap,next_gap_context)
        parameters=[projector.weight,projector.bias,head.weight]
        if head.bias is not None:parameters.append(head.bias)
        parameters=[value.detach().to(anchors).clone().requires_grad_(True) for value in parameters]
        support=support_features.detach().to(anchors)
        sy=support_labels.to(device=anchors.device,dtype=torch.long)
        z=F.linear(support,parameters[0],parameters[1])
        logits=F.linear(z,parameters[2],parameters[3] if len(parameters)>3 else None)
        objective=F.cross_entropy(logits,sy)
        if mse_weight:
            objective=objective+float(mse_weight)*F.mse_loss(z,anchors[sy])
        if mcl_weight:
            differences=z[:,None,:]-anchors[None,:,:]
            distances=(differences.square().sum(dim=2)+1e-12).sqrt()
            distances=add_mcl_gap(distances,sy,anchors,mcl_gap,mcl_gap_class_cap)
            objective=objective+float(mcl_weight)*F.cross_entropy(-distances,sy)
        gradients=torch.autograd.grad(objective,parameters,create_graph=create_graph)
        adapted=[value-float(inner_lr)*gradient for value,gradient in zip(parameters,gradients)]
        return adapted

def virtual_query_loss(support_features, support_labels, query_features, query_labels,
                       projector, head, anchors, inner_lr, mse_weight, mcl_weight,
                       mcl_gap, create_graph=False, mcl_gap_class_cap=False,
                       next_gap_context=None):
    """Functional support step and query CE; live parameters and RNG stay unchanged."""
    with torch.enable_grad():
        adapted=virtual_support_update(support_features,support_labels,projector,head,
            anchors,inner_lr,mse_weight,mcl_weight,mcl_gap,create_graph,
            mcl_gap_class_cap,next_gap_context)
        return adapted_query_loss(query_features,query_labels,adapted,anchors)


def adapted_query_loss(query_features,query_labels,adapted,anchors):
    """Evaluate an independent query against an already prepared virtual step."""
    query=query_features.detach().to(anchors)
    qy=query_labels.to(device=anchors.device,dtype=torch.long)
    query_z=F.linear(query,adapted[0],adapted[1])
    query_logits=F.linear(query_z,adapted[2],adapted[3] if len(adapted)>3 else None)
    return F.cross_entropy(query_logits,qy)


def candidate_direction_scores(support_features, support_labels, query_features,
                               query_labels, projector, head, current, candidate,
                               inner_lr, mse_weight, mcl_weight, mcl_gap,
                               return_gradient=False, path_max_weight=1.5,
                               estimate_query_noise=False, mcl_gap_class_cap=False,
                               next_gap_context=None):
    """Credit gain to anchor coordinates, rather than to query class labels.

    Probe the first half-step, completion to the candidate, and extra movement
    beyond the candidate separately. Their positions use the actual EMA-scaled
    proposal, so useful direction alone does not authorize a larger step.
    """
    if current.ndim!=2 or current.shape!=candidate.shape:
        raise ValueError('Current and candidate anchor matrices must match')
    if not torch.isfinite(current).all() or not torch.isfinite(candidate).all():
        raise FloatingPointError('Candidate trust received nonfinite anchors')
    if not math.isfinite(path_max_weight) or path_max_weight<=0:
        raise ValueError('Candidate probe path must have positive finite length')
    if estimate_query_noise and query_labels.numel()<2:
        raise ValueError('Two query observations are required to estimate query noise')
    def gradient_for_query(value,features,labels):
        with torch.enable_grad():
            bank=value.detach().clone().requires_grad_(True)
            loss=virtual_query_loss(support_features,support_labels,features,labels,
                projector,head,bank,inner_lr,mse_weight,mcl_weight,mcl_gap,create_graph=True,
                mcl_gap_class_cap=mcl_gap_class_cap,
                **({'next_gap_context':next_gap_context} if next_gap_context is not None else {}))
            if not torch.isfinite(loss):
                raise FloatingPointError('Candidate virtual query loss is nonfinite')
            gradient=torch.autograd.grad(loss,bank,allow_unused=True)[0]
            gradient=torch.zeros_like(bank) if gradient is None else gradient.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Candidate query gradient is nonfinite')
        return gradient,float(loss.detach())
    def query_gradient(value):
        if not estimate_query_noise:
            gradient,loss=gradient_for_query(value,query_features,query_labels)
            return gradient,loss,None
        split=query_labels.numel()//2
        first,first_loss=gradient_for_query(value,query_features[:split],query_labels[:split])
        second,second_loss=gradient_for_query(value,query_features[split:],query_labels[split:])
        fraction=split/query_labels.numel()
        scale=(fraction*first.double().square().sum()+(1-fraction)*second.double().square().sum()).sqrt()
        return (fraction*first+(1-fraction)*second,
                fraction*first_loss+(1-fraction)*second_loss,
                (first.double()-second.double(),fraction*(1-fraction),scale))
    if path_max_weight<=1:raise ValueError('Amplification probe must extend beyond the candidate')
    gradients=[];losses=[];query_noise=[]
    widths=(.5,.5,float(path_max_weight)-1)
    for fraction in (.25,.75,1+(float(path_max_weight)-1)/2):
        value=current.detach()+fraction*(candidate.detach()-current.detach())
        gradient,loss,noise=query_gradient(value)
        gradients.append(gradient.double());losses.append(loss);query_noise.append(noise)
    direction=candidate.detach().double()-current.detach().double()
    grad=sum(width*value for width,value in zip(widths,gradients))/float(path_max_weight)
    numerator=-(grad*direction).sum(dim=1)
    integrated=anchor_coordinate_direction_scores(grad,direction)
    scores=integrated['scores'];valid=integrated['valid']
    segment_scores=[];segment_valid=[];segment_gains=[]
    segment_cosines=[];segment_relative_norms=[]
    segment_noise_variances=[]
    for width,value,noise in zip(widths,gradients,query_noise):
        gain=-(value*direction).sum(dim=1)
        evidence=anchor_coordinate_direction_scores(value,direction)
        if noise is not None:
            difference,factor,scale=noise
            denominator=scale*direction.norm(dim=1)
            score=torch.zeros_like(denominator)
            sensitivity=torch.zeros_like(denominator)
            positive_scale=denominator>0
            score[positive_scale]=gain[positive_scale]/denominator[positive_scale]
            sensitivity[positive_scale]=(-(difference*direction).sum(dim=1)[positive_scale]
                                         /denominator[positive_scale])
            # A finite zero derivative is observed zero evidence. It must not
            # be treated as an unobserved client in a full-participation cohort.
            evidence['scores']=score.clamp(-1,1)
            evidence['valid']=direction.norm(dim=1)>0
            evidence['relative_gradient_norm']=value.norm(dim=1)/scale.clamp_min(1e-300)
            segment_noise_variances.append(factor*sensitivity.square())
        segment_scores.append(evidence['scores']);segment_valid.append(evidence['valid'])
        segment_gains.append(width*gain)
        segment_cosines.append(evidence['class_cosine'])
        segment_relative_norms.append(evidence['relative_gradient_norm'])
    result={'scores':scores.detach(),'valid':valid.detach(),
            'segment_scores':torch.stack(segment_scores),'segment_valid':torch.stack(segment_valid),
            'segment_class_cosines':torch.stack(segment_cosines),
            'segment_relative_gradient_norms':torch.stack(segment_relative_norms),
            'segment_gains':torch.stack(segment_gains),
            'directional_gain':float(path_max_weight)*numerator.detach(),'query_loss':sum(losses)/len(losses),
            'probe_path_max_weight':float(path_max_weight),'quadrature_nodes':3,
            'query_samples':int(query_labels.numel()),'mcl_gap':float(mcl_gap),
            'mcl_gap_class_cap':bool(mcl_gap_class_cap),
            'agreement_in_virtual_update':False,
            'score_normalization':'per_client_anchor_bank_gradient_frobenius',
            'within_client_gradient_magnitude_preserved':True}
    if estimate_query_noise:
        result['segment_query_noise_variance']=torch.stack(segment_noise_variances)
        result['score_normalization']='per_client_query_half_gradient_rms'
        result['query_noise_scope']='Two disjoint query halves, conditional on the same virtual support batch and pooled half-gradient RMS'
        result['support_noise_separately_estimated']=False
    if return_gradient:
        result['anchor_gradient']=query_gradient(current)[0]
        result['integrated_anchor_gradient']=grad.detach()
    return result


@torch.no_grad()
def candidate_confidence_update(scores, valid, client_weights, previous_mean=None,
                                previous_second=None, previous_count=None,
                                ema=.25, max_weight=1.5):
    """Turn reliable positive update evidence into weights below or above one.

    Evidence is never centered across classes or normalized to mean one.
    Repeated rounds do not manufacture additional independent clients.
    """
    if not math.isfinite(ema) or not 0<ema<=1 or not math.isfinite(max_weight) or max_weight<=1:
        raise ValueError('Candidate EMA must be in (0,1] and maximum weight must exceed one')
    if scores.ndim!=2 or valid.shape!=scores.shape or scores.size(1)==0:
        raise ValueError('Candidate scores and masks must have client by class shape')
    if client_weights.shape!=(scores.size(0),):
        raise ValueError('One weight is required per client')
    values=scores.double();mask=valid.bool()
    weights=client_weights.to(values)
    if not torch.isfinite(values[mask]).all() or (values[mask].abs()>1+1e-6).any():
        raise ValueError('Valid direction scores must be finite and within [-1,1]')
    if not torch.isfinite(weights).all() or (weights<0).any():
        raise ValueError('Client weights must be finite and nonnegative')
    weighted=mask.to(values)*weights[:,None]
    total=weighted.sum(dim=0)
    count=total.square()/weighted.square().sum(dim=0).clamp_min(1e-30)
    present=total>0
    safe_values=torch.where(mask,values,torch.zeros_like(values))
    mean=(weighted*safe_values).sum(dim=0)/total.clamp_min(1e-30)
    second=(weighted*safe_values.square()).sum(dim=0)/total.clamp_min(1e-30)
    old_mean=torch.zeros_like(mean) if previous_mean is None else previous_mean.to(mean)
    old_second=torch.zeros_like(second) if previous_second is None else previous_second.to(second)
    old_count=torch.zeros_like(count) if previous_count is None else previous_count.to(count)
    seen=present & (old_count>0)
    updated_mean=torch.where(present,mean,old_mean)
    updated_second=torch.where(present,second,old_second)
    updated_count=torch.where(present,count,old_count)
    updated_mean[seen]=(1-ema)*old_mean[seen]+ema*mean[seen]
    updated_second[seen]=(1-ema)*old_second[seen]+ema*second[seen]
    updated_count[seen]=(1-ema)*old_count[seen]+ema*count[seen]
    effective_count=torch.minimum(updated_count,count)
    variance=(updated_second-updated_mean.square()).clamp_min(0)
    variance=torch.where(effective_count>1,
        variance*effective_count/(effective_count-1).clamp_min(1e-30),torch.ones_like(variance))
    standard_error=(variance/effective_count.clamp_min(1)).sqrt()
    positive=updated_mean.clamp_min(0)
    confidence=(effective_count/(effective_count+1))*positive/(positive+standard_error+1e-12)
    confidence=torch.where(present,confidence,torch.zeros_like(confidence))
    return {'mean':updated_mean,'second':updated_second,'count':updated_count,
            'effective_clients':effective_count,'standard_error':standard_error,
            'confidence':confidence,
            'summary':{'signal':'heldout_train_query_directional_gain','class_mean_normalization':False,
                'max_weight':float(max_weight),'ema':float(ema),
                'classes_with_evidence':int(present.sum()),
                'mean_confidence':float(confidence.mean()),
                'mean_effective_clients':float(effective_count.mean())}}


@torch.no_grad()
def combine_candidate_step_evidence(estimates,max_weight=1.5):
    """Amplification needs reliable incremental gain beyond the full candidate."""
    if len(estimates)!=3 or not math.isfinite(max_weight) or max_weight<=1:
        raise ValueError('Three step segments and an amplification factor above one are required')
    first,second,extra=estimates
    base=.5*first['confidence']*(1+second['confidence'])
    count=extra['effective_clients']
    mean=extra['mean'];error=extra['standard_error']
    # Positive total gain is insufficient: penalize uncertainty in EXTRA gain.
    negative_mass=extra.get('negative_mass',torch.zeros_like(mean))
    reliable_extra=(count/(count+1))*(mean-2*error-negative_mass).clamp_min(0)/(mean.clamp_min(0)+error+negative_mass+1e-12)
    if 'current_mean' in extra:
        current_mean=extra['current_mean'];current_error=extra['current_standard_error']
        current_downside=extra['current_negative_mass']
        current_extra=(count/(count+1))*(current_mean-2*current_error-current_downside).clamp_min(0)/(
            current_mean.clamp_min(0)+current_error+current_downside+1e-12)
        reliable_extra=torch.minimum(reliable_extra,current_extra)
    reliable_extra=torch.where(count>0,reliable_extra,torch.zeros_like(reliable_extra))
    # Each interval adds its own movement. Extra gain measured beyond the
    # full candidate must not multiply a partial first interval and push it
    # into a harmful middle interval that had no supporting evidence.
    completion_confidence=first['confidence']*second['confidence']
    extra_step=(float(max_weight)-1)*completion_confidence*reliable_extra
    coefficient=base+extra_step
    return {'base_coefficient':base,'amplification_confidence':reliable_extra,
            'coefficient':coefficient,'summary':{
                'signal':'heldout_train_query_step_segment_gain','class_mean_normalization':False,
                'max_weight':float(max_weight),'extra_gain_uncertainty_penalty':2.,
                'separate_negative_effect_penalty':'negative_mass' in extra,
                'current_extra_evidence_required':'current_mean' in extra,
                'extra_step_requires_prefix_confidence':True,
                'step_composition':'additive_sequential_intervals',
                'mean_extra_step_coefficient':float(extra_step.mean()),
                'amplified_classes':int((coefficient>1+1e-6).sum()),
                'attenuated_classes':int((coefficient<1-1e-6).sum()),
                'mean_coefficient':float(coefficient.mean()),'mean_base_coefficient':float(base.mean()),
                'mean_amplification_confidence':float(reliable_extra.mean())}}
