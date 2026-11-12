"""Finite, paired train-query comparisons for the original RAAR beta family."""
import math

import torch

from utils.anchor_candidate_probe import one_step_anchor_bank_losses
from utils.raar_next_gap import candidate_next_gap_kwargs


class CachedRAARQueries:
    def __init__(self,clients,round_idx,stable,candidate,population_clients,query_prefix):
        if query_prefix not in {'allocation','diagnostic'}:
            raise ValueError('Choose cached acceptance batch2 or diagnostic batch4')
        self.round_idx=int(round_idx)
        self.stable=stable.detach();self.candidate=candidate.detach()
        self.population=int(population_clients)
        self.prefix=query_prefix
        self.feature_key=query_prefix+'_query_features'
        self.label_key=query_prefix+'_query_labels'
        self.expected=len(clients)
        self.clients=[]
        for client in clients:
            cache=getattr(client,'anchor_candidate_evidence',None)
            if cache is not None and cache.get('round')==self.round_idx and self.feature_key in cache and self.label_key in cache:
                self.clients.append(client)
        if self.population<self.expected:
            raise ValueError('The population must include the selected client cohort')
        self.cache={}
        self.bank_evaluations=0

    def _losses(self,beta):
        key=tuple(beta.detach().double().cpu().tolist())
        if key not in self.cache:
            weights=beta.detach().to(self.stable)[:,None]
            bank=self.stable+weights*(self.candidate-self.stable)
            bank=torch.where(weights==1,self.candidate,bank)
            losses=[]
            for client in self.clients:
                evidence=client.anchor_candidate_evidence
                device=client.model.shared_projector.weight.device
                result=one_step_anchor_bank_losses(
                    evidence['support_features'].to(device),evidence['support_labels'].to(device),
                    evidence[self.feature_key].to(device),evidence[self.label_key].to(device),
                    client.model.shared_projector,client.model.shared_head,{'bank':bank},
                    client.utility_inner_lr,client.lamda,client.mcl,evidence['mcl_gap'],
                    **({'mcl_gap_class_cap':True} if evidence.get('mcl_gap_class_cap',False) else {}),
                    **candidate_next_gap_kwargs(client,evidence))
                losses.append(result['bank'].detach())
            self.cache[key]=losses
            self.bank_evaluations+=1
        return self.cache[key]

    def evaluate(self,trial,reference):
        if not self.clients or len(self.clients)!=self.expected:
            return {'observed_clients':0,'available_clients':len(self.clients),
                'mean_gain':None,'standard_error':None,'reason':'Incomplete current-round cached queries'}
        if torch.equal(trial,reference):
            return {'observed_clients':len(self.clients),
                'query_samples':sum(client.anchor_candidate_evidence[self.label_key].numel() for client in self.clients),
                'mean_gain':0.,'standard_error':0.,'within_query_mean_variance':0.,
                'client_sampling_variance':0.,'numerical_tolerance':0.,
                'query_batch':2 if self.prefix=='allocation' else 4,'analytic_equal_banks':True,
                'uses_test_data':False,'model_parameters_changed':False}
        trial_losses=self._losses(trial)
        reference_losses=self._losses(reference)
        pairs=[]
        precision_floor=0.
        for client,left,right in zip(self.clients,trial_losses,reference_losses):
            gain=right.double()-left.double()
            n=gain.numel()
            if n<2 or not torch.isfinite(gain).all():
                return {'observed_clients':0,'mean_gain':None,'standard_error':None,
                        'reason':'Too few finite paired queries'}
            pairs.append({'client':int(client.id),'n':n,'mean':float(gain.mean()),
                          'variance_of_mean':float(gain.var(unbiased=True)/n)})
            scale=max(1.,float(left.abs().mean()),float(right.abs().mean()))
            precision_floor=max(precision_floor,4*torch.finfo(left.dtype).eps*scale)
        total=sum(row['n'] for row in pairs)
        shares=[row['n']/total for row in pairs]
        mean=sum(share*row['mean'] for share,row in zip(shares,pairs))
        within=sum(share*share*row['variance_of_mean'] for share,row in zip(shares,pairs))
        effective=1/sum(share*share for share in shares)
        between=sum(share*(row['mean']-mean)**2 for share,row in zip(shares,pairs))
        between=between*effective/(effective-1) if effective>1 else 1.
        sampling=(1-len(pairs)/self.population)*between/effective
        return {'observed_clients':len(pairs),'query_samples':total,'mean_gain':mean,
            'standard_error':math.sqrt(max(0.,within+sampling)),
            'within_query_mean_variance':within,'client_sampling_variance':sampling,
            'numerical_tolerance':precision_floor,'query_batch':2 if self.prefix=='allocation' else 4,
            'uses_test_data':False,'model_parameters_changed':False,'clients':pairs,
            'scope':'Conditional paired-query uncertainty. Support-batch noise and repeated-comparison selection are not calibrated.'}
