"""Finite, paired train-query comparisons for the original RAAR beta family."""
import math

import torch

from utils.anchor_candidate_probe import one_step_anchor_bank_losses, prepare_anchor_bank_probe
from utils.raar_next_gap import candidate_next_gap_kwargs
from utils.frozen_anchor_geometry import FrozenAnchorGeometry


class CachedRAARQueries:
    """Round-scoped oracle; client models and evidence remain fixed during use."""

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
        self.prepared_clients={}
        self.bank_evaluations=0

    def _losses(self,beta):
        key=tuple(beta.detach().double().cpu().tolist())
        if key not in self.cache:
            weights=beta.detach().to(self.stable)[:,None]
            bank=self.stable+weights*(self.candidate-self.stable)
            bank=torch.where(weights==1,self.candidate,bank)
            losses=[]
            geometries={}
            for client in self.clients:
                evidence=client.anchor_candidate_evidence
                identity=id(client)
                if identity not in self.prepared_clients:
                    self.prepared_clients[identity]=prepare_anchor_bank_probe(
                        evidence['support_features'],evidence['support_labels'],
                        evidence[self.feature_key],evidence[self.label_key],
                        client.model.shared_projector,client.model.shared_head)
                prepared=self.prepared_clients[identity]
                geometry=None
                if client.mcl and (evidence.get('mcl_gap_class_cap',False)
                                   or getattr(client,'candidate_next_gap',False)):
                    device=prepared['params'][0].device
                    if device not in geometries:
                        geometries[device]=FrozenAnchorGeometry(bank.detach().to(device=device,dtype=torch.float32))
                    geometry=geometries[device]
                result=one_step_anchor_bank_losses(
                    prepared['support_features'],prepared['support_labels'],
                    prepared['query_features'],prepared['query_labels'],
                    client.model.shared_projector,client.model.shared_head,{'bank':bank},
                    client.utility_inner_lr,client.lamda,client.mcl,evidence['mcl_gap'],
                    **({'mcl_gap_class_cap':True} if evidence.get('mcl_gap_class_cap',False) else {}),
                    **candidate_next_gap_kwargs(client,evidence), _prepared=prepared,
                    _bank_geometry=geometry)
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
        # Reduce each client on its original device/dtype, then transfer all
        # scalar statistics once per CUDA device. Python cohort aggregation below
        # retains its original client order and floating-point arithmetic.
        statistics=[]
        transfers={}
        for client,left,right in zip(self.clients,trial_losses,reference_losses):
            gain=right.double()-left.double()
            n=gain.numel()
            if n<2:
                return {'observed_clients':0,'mean_gain':None,'standard_error':None,
                        'reason':'Too few finite paired queries'}
            finite=torch.isfinite(gain).all()
            if gain.device.type=='cpu' and not bool(finite):
                return {'observed_clients':0,'mean_gain':None,'standard_error':None,
                        'reason':'Too few finite paired queries'}
            values=(gain.mean(),gain.var(unbiased=True)/n,left.abs().mean(),right.abs().mean())
            index=len(statistics)
            if gain.device.type=='cpu':
                scalars=[1.,*(float(value) for value in values)]
            else:
                scalars=None
                packed=torch.stack((finite.double(),*(value.double() for value in values)))
                transfers.setdefault(gain.device,[]).append((index,packed))
            statistics.append((int(client.id),n,torch.finfo(left.dtype).eps,scalars))
        for entries in transfers.values():
            values=torch.stack([packed for _,packed in entries]).cpu().tolist()
            for (index,_),scalars in zip(entries,values):
                client_id,n,epsilon,_=statistics[index]
                statistics[index]=(client_id,n,epsilon,scalars)
        pairs=[]
        precision_floor=0.
        for client_id,n,epsilon,values in statistics:
            finite,mean_gain,variance,left_scale,right_scale=values
            if not finite:
                return {'observed_clients':0,'mean_gain':None,'standard_error':None,
                        'reason':'Too few finite paired queries'}
            pairs.append({'client':client_id,'n':n,'mean':mean_gain,
                          'variance_of_mean':variance})
            scale=max(1.,left_scale,right_scale)
            precision_floor=max(precision_floor,4*epsilon*scale)
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
