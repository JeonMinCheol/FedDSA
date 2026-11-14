"""Train-only, read-only one-step validation of alternative anchor banks."""
import itertools

import torch
import torch.nn.functional as F
from utils.mcl_gap import add_mcl_gap
from utils.raar_next_gap import candidate_next_gap_kwargs


@torch.no_grad()
def same_budget_anchor_banks(current,candidate,applied,adaptive_weights,observed,private_seed):
    """Compare allocation rules at the current state's identical total movement."""
    from utils.anchor_budget_control import budget_matched_anchor_update
    target=float((applied-current).norm(dim=1).mean())
    banks={'current':current.detach().clone(),'applied':applied.detach().clone()}
    budgets={}
    for name,mode in (('aligned','budget_aligned'),('uniform','budget_uniform'),('shuffled','budget_shuffled')):
        bank,_,summary=budget_matched_anchor_update(current,candidate,adaptive_weights,
                                                   observed,target,mode,private_seed)
        banks[name]=bank
        budgets[name]=summary
    return banks,budgets


def prepare_anchor_bank_probe(support_features, support_labels, query_features,
                              query_labels, projector, head):
    """Snapshot fixed inputs/leaf parameters for one cached round's bank family.

    No support graph is retained. Every bank still builds its own forward and
    backward graph; parameter leaves are read-only and autograd.grad does not
    accumulate into their .grad fields.
    """
    if not isinstance(projector,torch.nn.Linear) or not isinstance(head,torch.nn.Linear):
        raise ValueError('The candidate probe requires linear projector and classifier')
    if projector.bias is None:
        raise ValueError('The candidate probe requires a biased projector')
    device=projector.weight.device
    params=[projector.weight.detach().float().clone().requires_grad_(True),
            projector.bias.detach().float().clone().requires_grad_(True),
            head.weight.detach().float().clone().requires_grad_(True)]
    if head.bias is not None:
        params.append(head.bias.detach().float().clone().requires_grad_(True))
    return dict(params=params,
        support_features=support_features.detach().to(device).float(),
        support_labels=support_labels.to(device),
        query_features=query_features.detach().to(device).float(),
        query_labels=query_labels.to(device))


def one_step_anchor_bank_losses(support_features, support_labels, query_features, query_labels,
                                projector, head, anchor_banks, inner_lr, mse_weight,
                                mcl_weight, mcl_gap, mcl_gap_class_cap=False,
                                next_gap_context=None, _prepared=None, _bank_geometry=None):
    """Change only the anchor bank, then evaluate a virtual update on the same query.

    Parameter clones are independent leaves: neither parameters nor .grad fields
    on the trained model are modified. Agreement is intentionally absent and the
    MCL gap is held fixed across alternatives.
    """
    prepared=(_prepared if _prepared is not None else prepare_anchor_bank_probe(
        support_features,support_labels,query_features,query_labels,projector,head))
    params=prepared['params']
    support_features=prepared['support_features'];support_labels=prepared['support_labels']
    query_features=prepared['query_features'];query_labels=prepared['query_labels']
    losses={}
    for name,bank in anchor_banks.items():
        with torch.enable_grad():
            projected=F.linear(support_features,params[0],params[1])
            logits=F.linear(projected,params[2],params[3] if len(params)>3 else None)
            bank=bank.detach().to(device=projected.device,dtype=torch.float32)
            gap=mcl_gap
            if next_gap_context is not None:
                if _bank_geometry is None:
                    from utils.raar_next_gap import resolve_next_gap
                    gap=resolve_next_gap(bank,mcl_gap,next_gap_context)
                else:
                    gap=_bank_geometry.next_gap(mcl_gap,next_gap_context)
            objective=F.cross_entropy(logits,support_labels)
            if mse_weight:
                objective=objective+float(mse_weight)*F.mse_loss(projected,bank[support_labels])
            if mcl_weight:
                distances=torch.cdist(projected,bank,p=2)
                distances=(add_mcl_gap(distances,support_labels,bank,gap,mcl_gap_class_cap)
                    if _bank_geometry is None else
                    _bank_geometry.add_gap(distances,support_labels,gap,mcl_gap_class_cap))
                objective=objective+float(mcl_weight)*F.cross_entropy(-distances,support_labels)
            gradients=torch.autograd.grad(objective,params)
            adapted=[(value-float(inner_lr)*gradient).detach() for value,gradient in zip(params,gradients)]
        with torch.no_grad():
            query_projected=F.linear(query_features,adapted[0],adapted[1])
            query_logits=F.linear(query_projected,adapted[2],adapted[3] if len(adapted)>3 else None)
            per_sample=F.cross_entropy(query_logits,query_labels,reduction='none')
            if not torch.isfinite(per_sample).all():
                raise ValueError('Candidate probe generated nonfinite query losses')
            losses[name]=per_sample.detach()
    return losses


def probe_client_anchor_banks(client, anchor_banks, private_seed):
    """Probe two disjoint shuffled batches of client training data and restore state."""
    model=getattr(client,'model',None)
    if model is None:
        raise ValueError('The client has no model available for candidate validation')
    original_device=next(model.parameters()).device
    device=torch.device(client.device)
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    training_flags=[(module,module.training) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=devices):
            # Seed only the CPU data-loader RNG; avoid changing unrelated GPU generators.
            generator=torch.Generator(device='cpu').manual_seed(int(private_seed)%(2**63-1))
            torch.set_rng_state(generator.get_state())
            batches=list(itertools.islice(client.load_train_data(drop_last=False),2))
            if len(batches)<2:
                return {'client_id':int(client.id),'available':False,'reason':'Fewer than two training batches'}
            inputs=[]
            for x,y in batches:
                if isinstance(x,(list,tuple)):
                    x=x[0]
                inputs.append((x.to(device),y.to(device).long().reshape(-1)))
            model.to(device)
            model.eval()
            with torch.no_grad():
                support_features=model.extract_backbone(inputs[0][0]).detach().float()
                query_features=model.extract_backbone(inputs[1][0]).detach().float()
            losses=one_step_anchor_bank_losses(
                support_features,inputs[0][1],query_features,inputs[1][1],
                model.shared_projector,model.shared_head,anchor_banks,
                client.utility_inner_lr,client.lamda,client.mcl,client.mcl_gap_value(),
                **({'mcl_gap_class_cap':True} if getattr(client,'mcl_gap_class_cap',False) else {}),
                **candidate_next_gap_kwargs(client,{
                    'round':int(getattr(client,'current_round',0)),
                    'candidate_next_gap':True,
                    'next_local_gap':client.nearest_gap_from_dict(client.protos)
                        if getattr(client,'candidate_next_gap',False) else 0.,
                    'next_gap_mode':getattr(client,'mcl_gap_mode','min')}))
            labels=inputs[1][1]
            count=torch.bincount(labels,minlength=client.num_classes).float()
            result={'client_id':int(client.id),'available':True,'support_samples':int(inputs[0][1].numel()),
                    'query_samples':int(labels.numel()),'query_class_count':count.cpu().tolist(),
                    'mcl_gap':float(client.mcl_gap_value()),'agreement_in_virtual_step':False,
                    'banks':{}}
            reference=losses['current']
            for name,values in losses.items():
                gain=reference-values
                sums=torch.zeros_like(count)
                sums.index_add_(0,labels,gain)
                result['banks'][name]={
                    'mean_query_loss':float(values.mean()),'mean_gain_vs_current_bank':float(gain.mean()),
                    'class_gain_sum':sums.cpu().tolist()}
            return result
    finally:
        model.to(original_device)
        for module,flag in training_flags:
            module.training=flag


def aggregate_candidate_probes(records, num_classes):
    """Aggregate class gains with query sample counts, without treating classes as replicates."""
    available=[record for record in records if record.get('available')]
    if not available:
        return {'available_clients':0,'query_samples':0,'banks':{}}
    count=[sum(record['query_class_count'][c] for record in available) for c in range(num_classes)]
    total=sum(count)
    result={'available_clients':len(available),'query_samples':int(total),
            'observed_query_classes':sum(value>0 for value in count),
            'query_class_count':count,'banks':{}}
    for name in available[0]['banks']:
        sums=[sum(record['banks'][name]['class_gain_sum'][c] for record in available) for c in range(num_classes)]
        result['banks'][name]={
            'mean_gain_vs_current_bank':sum(sums)/total,
            'class_gain_vs_current_bank':[value/n if n>0 else None for value,n in zip(sums,count)]}
    return result
